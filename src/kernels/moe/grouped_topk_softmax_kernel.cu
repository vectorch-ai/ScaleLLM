// Adapted from https://github.com/NVIDIA/FasterTransformer
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cutlass/array.h>
#include <torch/torch.h>

#include <algorithm>
#include <cfloat>

namespace llm::kernel {
namespace {
constexpr int WARP_SIZE = 32;

template <int EXPERTS, int ELTS_PER_THR, int WARPS_PER_CTA, int BYTES_PER_LDG>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE) __global__
    void topk_softmax_kernel(const float* logits,
                             float* weights,
                             int* indices,
                             const int64_t n_tokens,
                             const int topk) {
  static_assert(ELTS_PER_THR == (ELTS_PER_THR & -ELTS_PER_THR),
                "ELTS_PER_THR must be power of 2");
  static_assert(EXPERTS == (EXPERTS & -EXPERTS),
                "NUM_EXPERTS must be power of 2");
  static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG),
                "BYTES_PER_LDG must be power of 2");
  static_assert(BYTES_PER_LDG <= 16, "BYTES_PER_LDG must be leq 16");

  // Number of bytes each thread pulls in per load
  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(float);
  static constexpr int ELTS_PER_TOKEN = EXPERTS;
  static constexpr int THRS_PER_TOKEN = ELTS_PER_TOKEN / ELTS_PER_THR;
  static constexpr int LDGS_PER_THR = ELTS_PER_THR / ELTS_PER_LDG;

  static_assert(
      ELTS_PER_THR % ELTS_PER_LDG == 0,
      "The elements per thread must be a multiple of the elements per ldg");

  static_assert(THRS_PER_TOKEN == (THRS_PER_TOKEN & -THRS_PER_TOKEN),
                "THREADS_PER_TOKEN must be power of 2");
  static_assert(THRS_PER_TOKEN <= WARP_SIZE,
                "THREADS_PER_TOKEN can be at most warp size");
  static_assert(WARP_SIZE % THRS_PER_TOKEN == 0,
                "THREADS_PER_TOKEN must cleanly divide warp size");

  static constexpr int ELTS_PER_WARP = WARP_SIZE * ELTS_PER_THR;
  static constexpr int TOKENS_PER_WARP = ELTS_PER_WARP / ELTS_PER_TOKEN;
  static constexpr int TOKENS_PER_CTA = WARPS_PER_CTA * TOKENS_PER_WARP;

  static_assert(
      ELTS_PER_WARP % ELTS_PER_TOKEN == 0,
      "The elts per token must cleanly divide the total elt per warp");

  // calculate token index for this thread.
  const int64_t cta_base = blockIdx.x * TOKENS_PER_CTA;
  const int64_t warp_base = cta_base + (threadIdx.y * TOKENS_PER_WARP);
  const int64_t token_idx = warp_base + (threadIdx.x / THRS_PER_TOKEN);

  // out of bounds, early exit.
  if (token_idx >= n_tokens) {
    return;
  }

  // calculate the base address for this thread.
  // logits [n_tokens, LDGS_PER_THR, THRS_PER_TOKEN, ELTS_PER_LDG]
  const float* logits_ptr = logits + (token_idx * ELTS_PER_TOKEN);
  const int thread_idx_in_group = threadIdx.x % THRS_PER_TOKEN;
  const int col_base = thread_idx_in_group * ELTS_PER_LDG;
  const float* thread_read_ptr = logits_ptr + col_base;

  using AccessType = cutlass::AlignedArray<float, ELTS_PER_LDG>;

  // load data from global mem to register.
  // row_chunk: [LDGS_PER_THR, ELTS_PER_LDG]
  cutlass::Array<float, ELTS_PER_THR> row_chunk;
  AccessType* row_chunk_vec_ptr = reinterpret_cast<AccessType*>(&row_chunk);
  const AccessType* vec_thread_read_ptr =
      reinterpret_cast<AccessType const*>(thread_read_ptr);
  CUTE_UNROLL
  for (int i = 0; i < LDGS_PER_THR; ++i) {
    row_chunk_vec_ptr[i] = vec_thread_read_ptr[i * THRS_PER_TOKEN];
  }

  // 1. perform a softmax for the token.
  // max reduce within the thread
  float row_max = row_chunk[0];
  CUTE_UNROLL
  for (int i = 1; i < ELTS_PER_THR; ++i) {
    row_max = max(row_max, row_chunk[i]);
  }

  // max reduce within threads for the token.
  CUTE_UNROLL
  for (int mask = THRS_PER_TOKEN / 2; mask > 0; mask /= 2) {
    row_max = max(row_max,
                  __shfl_xor_sync(0xFFFFFFFF, row_max, mask, THRS_PER_TOKEN));
  }

  // local sum within thread
  float row_sum = 0;
  CUTE_UNROLL
  for (int i = 0; i < ELTS_PER_THR; ++i) {
    const float val = expf(row_chunk[i] - row_max);
    row_chunk[i] = val;
    row_sum += val;
  }

  // sum reduce within threads for the token.
  CUTE_UNROLL
  for (int mask = THRS_PER_TOKEN / 2; mask > 0; mask /= 2) {
    row_sum += __shfl_xor_sync(0xFFFFFFFF, row_sum, mask, THRS_PER_TOKEN);
  }

  const float reciprocal_row_sum = 1.f / row_sum;
  CUTE_UNROLL
  for (int ii = 0; ii < ELTS_PER_THR; ++ii) {
    row_chunk[ii] = row_chunk[ii] * reciprocal_row_sum;
  }

  // 2. find topk experts for the token.
  int start_col = col_base;
  static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THRS_PER_TOKEN;

  for (int k_idx = 0; k_idx < topk; ++k_idx) {
    // local argmax within thread
    float max_val = row_chunk[0];
    int max_col = start_col;

    int col = start_col;
    CUTE_UNROLL
    for (int ldg = 0; ldg < LDGS_PER_THR; ++ldg) {
      CUTE_UNROLL
      for (int j = 0; j < ELTS_PER_LDG; ++j) {
        const float val = row_chunk[ldg * ELTS_PER_LDG + j];
        if (val > max_val) {
          max_val = val;
          max_col = col + j;
        }
      }
      col += COLS_PER_GROUP_LDG;
    }

    // a argmax reduction within threads for the token.
    CUTE_UNROLL
    for (int mask = THRS_PER_TOKEN / 2; mask > 0; mask /= 2) {
      float other_max =
          __shfl_xor_sync(0xFFFFFFFF, max_val, mask, THRS_PER_TOKEN);
      int other_col =
          __shfl_xor_sync(0xFFFFFFFF, max_col, mask, THRS_PER_TOKEN);

      // lower indices to "win" to break ties.
      if (other_max > max_val ||
          (other_max == max_val && other_col < max_col)) {
        max_val = other_max;
        max_col = other_col;
      }
    }

    // write the max to global memory.
    if (thread_idx_in_group == 0) {
      const int64_t idx = topk * token_idx + k_idx;
      weights[idx] = max_val;
      indices[idx] = max_col;
    }

    // mask out current max value
    if (k_idx + 1 < topk) {
      const int thread_idx_to_clear = (max_col / ELTS_PER_LDG) % THRS_PER_TOKEN;

      // Only the thread in the group which produced the max will reset the
      // "winning" value to -inf.
      if (thread_idx_in_group == thread_idx_to_clear) {
        const int ldg_idx = max_col / COLS_PER_GROUP_LDG;
        const int offset = max_col % ELTS_PER_LDG;
        // Safe to set to any negative value since row_chunk values must be
        // between 0 and 1.
        row_chunk[ldg_idx * ELTS_PER_LDG + offset] = -10000.f;
      }
    }
  }
}

template <int EXPERTS, int WARPS_PER_TB>
void launch_topk_softmax_kernel_(const float* logits,
                                 float* weights,
                                 int* indices,
                                 const int64_t n_tokens,
                                 const int topk,
                                 cudaStream_t stream) {
  static constexpr std::size_t MAX_BYTES_PER_LDG = 16;

  // up to 16 bytes per ldg
  static constexpr int BYTES_PER_LDG =
      std::min(MAX_BYTES_PER_LDG, sizeof(float) * EXPERTS);
  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(float);
  static_assert(EXPERTS / (ELTS_PER_LDG * WARP_SIZE) == 0 ||
                EXPERTS % (ELTS_PER_LDG * WARP_SIZE) == 0);

  static constexpr int LDGS_PER_THR =
      std::max(1, EXPERTS / (ELTS_PER_LDG * WARP_SIZE));
  static constexpr int ELTS_PER_THR = LDGS_PER_THR * ELTS_PER_LDG;
  static constexpr int THRS_PER_TOKEN = EXPERTS / ELTS_PER_THR;
  static constexpr int TOKENS_PER_WARP = WARP_SIZE / THRS_PER_TOKEN;

  const int64_t num_warps = (n_tokens + TOKENS_PER_WARP - 1) / TOKENS_PER_WARP;
  const int64_t num_blocks = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB;

  dim3 block_dim(WARP_SIZE, WARPS_PER_TB);
  topk_softmax_kernel<EXPERTS, ELTS_PER_THR, WARPS_PER_TB, BYTES_PER_LDG>
      <<<num_blocks, block_dim, 0, stream>>>(
          logits, weights, indices, n_tokens, topk);
}

void launch_topk_softmax_kernel(const float* logits,
                                float* weights,
                                int* indices,
                                const int64_t n_tokens,
                                const int n_experts,
                                const int topk,
                                cudaStream_t stream) {
  static constexpr int WARPS_PER_TB = 4;

#define LAUNCH_TOPK_SOFTMAX_KERNEL(EXPERTS)           \
  launch_topk_softmax_kernel_<EXPERTS, WARPS_PER_TB>( \
      logits, weights, indices, n_tokens, topk, stream);

  switch (n_experts) {
    case 1: {
      LAUNCH_TOPK_SOFTMAX_KERNEL(1);
      break;
    }
    case 2: {
      LAUNCH_TOPK_SOFTMAX_KERNEL(2);
      break;
    }
    case 4: {
      LAUNCH_TOPK_SOFTMAX_KERNEL(4);
      break;
    }
    case 8: {
      LAUNCH_TOPK_SOFTMAX_KERNEL(8);
      break;
    }
    case 16: {
      LAUNCH_TOPK_SOFTMAX_KERNEL(16);
      break;
    }
    case 32: {
      LAUNCH_TOPK_SOFTMAX_KERNEL(32);
      break;
    }
    case 64: {
      LAUNCH_TOPK_SOFTMAX_KERNEL(64);
      break;
    }
    case 128: {
      LAUNCH_TOPK_SOFTMAX_KERNEL(128);
      break;
    }
    case 256: {
      LAUNCH_TOPK_SOFTMAX_KERNEL(256);
      break;
    }
    default: {
      CHECK(false) << "Unsupported number of experts: " << n_experts;
    }
  }
}

}  // namespace

void grouped_topk_softmax(
    const torch::Tensor& gating_logit,  // [n_tokens, n_experts]
    torch::Tensor& topk_weights,        // [n_tokens, topk]
    torch::Tensor& topk_indices         // [n_tokens, topk]
) {
  const int n_tokens = gating_logit.size(0);
  const int n_experts = gating_logit.size(1);
  const int topk = topk_weights.size(-1);

  const bool is_pow_2 =
      (n_experts != 0) && ((n_experts & (n_experts - 1)) == 0);
  CHECK(is_pow_2) << "Number of experts must be a power of 2, but got: "
                  << n_experts;

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_topk_softmax_kernel(gating_logit.const_data_ptr<float>(),
                             topk_weights.data_ptr<float>(),
                             topk_indices.data_ptr<int>(),
                             n_tokens,
                             n_experts,
                             topk,
                             stream);
}

}  // namespace llm::kernel