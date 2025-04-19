#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cutlass/array.h>
#include <torch/torch.h>

#include <algorithm>
#include <cfloat>

namespace llm::kernel {
namespace {
constexpr int WARP_SIZE = 32;

// template <int EXPERTS, int ELTS_PER_THR, int WARPS_PER_CTA, int
// BYTES_PER_LDG>
template <int EXPERTS, int THRS_PER_TOKEN, int WARPS_PER_CTA>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE) __global__
    void grouped_topk_sigmoid_kernel(
        const float* logits,  // [n_tokens, n_experts]
        const float* bias,    // [n_experts]
        float* weights,
        int* indices,
        const int64_t n_tokens,
        const int topk_group,
        const int topk,
        const float scaling_factor) {
  static_assert(EXPERTS == (EXPERTS & -EXPERTS),
                "NUM_EXPERTS must be power of 2");
  static constexpr int ELTS_PER_TOKEN = EXPERTS;
  // static constexpr int THRS_PER_TOKEN = ELTS_PER_TOKEN / ELTS_PER_THR;
  static constexpr int ELTS_PER_THR = EXPERTS / THRS_PER_TOKEN;

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
  const float* logits_ptr = logits + (token_idx * ELTS_PER_TOKEN);
  const int thread_idx_in_group = threadIdx.x % THRS_PER_TOKEN;
  const int col_base = thread_idx_in_group * ELTS_PER_THR;
  const float* logits_read_ptr = logits_ptr + col_base;
  const float* bias_read_ptr = bias + col_base;

  // load data from global mem to register.
  using ArrayType = cutlass::AlignedArray<float, ELTS_PER_THR>;
  ArrayType scores_chunk;
  ArrayType tmp_scores_chunk;
  // whether current thread/expert group is masked out.
  bool group_masked_out = false;

  CUTE_UNROLL
  for (int i = 0; i < ELTS_PER_THR; ++i) {
    scores_chunk[i] = logits_read_ptr[i];
    tmp_scores_chunk[i] = bias_read_ptr[i];
  }

  // 1. apply sigmoid
  CUTE_UNROLL
  for (int i = 0; i < ELTS_PER_THR; ++i) {
    scores_chunk[i] = 1.0f / (1.f + expf(-scores_chunk[i]));
    tmp_scores_chunk[i] += scores_chunk[i];
  }

  int start_col = col_base;
  // 2. select topk_group and mask out others.
  // THREADS_PER_ROW == num_expert_groups
  for (int k_idx = 0; k_idx < THRS_PER_TOKEN - topk_group; ++k_idx) {
    // top2 of each expert group
    // local max and second max within thread
    float max_val = -FLT_MAX;
    float second_max_val = -FLT_MAX;
    CUTE_UNROLL
    for (int i = 0; i < ELTS_PER_THR; ++i) {
      auto val = tmp_scores_chunk[i];
      if (val > max_val) {
        second_max_val = max_val;
        max_val = val;
      } else if (val > second_max_val) {
        second_max_val = val;
      }
    }

    // find the minimum of the sum of the top2
    auto min_sum = max_val + second_max_val;
    auto min_col = start_col;
    // a max reduction within threads for the token.
    CUTE_UNROLL
    for (int mask = THRS_PER_TOKEN / 2; mask > 0; mask /= 2) {
      float other_sum =
          __shfl_xor_sync(0xFFFFFFFF, max_val, mask, THRS_PER_TOKEN);
      int other_col =
          __shfl_xor_sync(0xFFFFFFFF, min_col, mask, THRS_PER_TOKEN);

      // higher indices to "win" to break ties.
      if (other_sum < min_sum ||
          (other_sum == min_sum && other_col > min_col)) {
        min_sum = other_sum;
        min_col = other_col;
      }
    }

    // mask out the expert group with the minimum sum of top2
    if (k_idx + 1 < THRS_PER_TOKEN - topk_group) {
      const int thread_idx_to_clear = min_col / ELTS_PER_TOKEN;
      if (thread_idx_in_group == thread_idx_to_clear) {
        group_masked_out = true;
      }
    }
  }

  // 3. find topk experts for the token.
  for (int k_idx = 0; k_idx < topk; ++k_idx) {
    // local argmax within thread
    int max_col = start_col;
    float max_val = -FLT_MAX;

    if (!group_masked_out) {
      max_val = tmp_scores_chunk[0];
      CUTE_UNROLL
      for (int i = 1; i < ELTS_PER_THR; ++i) {
        auto val = tmp_scores_chunk[i];
        if (val > max_val) {
          max_val = val;
          max_col = start_col + i;
        }
      }
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

    const auto col_in_thread = max_col % ELTS_PER_TOKEN;
    // write the max to global memory.
    if (thread_idx_in_group == 0) {
      const int64_t idx = (topk * token_idx) + k_idx;
      // fetch the original scores
      weights[idx] = scores_chunk[col_in_thread];
      indices[idx] = max_col;
    }

    // mask out current max value
    if (k_idx + 1 < topk) {
      const int thread_idx_to_clear = max_col / ELTS_PER_TOKEN;
      if (thread_idx_in_group == thread_idx_to_clear) {
        tmp_scores_chunk[col_in_thread] = -FLT_MAX;
      }
    }
  }

  // TODO: 4. apply the scaling factor
}

template <int EXPERTS, int EXPERTS_GROUPS, int WARPS_PER_CTA>
void launch_grouped_topk_sigmoid_kernel_(const float* logits,
                                         const float* bias,
                                         float* weights,
                                         int* indices,
                                         const int64_t n_tokens,
                                         const int topk_group,
                                         const int topk,
                                         const float scaling_factor,
                                         cudaStream_t stream) {
  // for simplicity, one expert group per thread.
  static constexpr int THRS_PER_TOKEN = EXPERTS_GROUPS;
  static constexpr int TOKENS_PER_WARP =
      std::max(1, WARP_SIZE / THRS_PER_TOKEN);

  const int64_t num_warps = (n_tokens + TOKENS_PER_WARP - 1) / TOKENS_PER_WARP;
  const int64_t num_blocks = (num_warps + WARPS_PER_CTA - 1) / WARPS_PER_CTA;

  dim3 block_dim(WARP_SIZE, WARPS_PER_CTA);
  grouped_topk_sigmoid_kernel<EXPERTS, THRS_PER_TOKEN, WARPS_PER_CTA>
      <<<num_blocks, block_dim, 0, stream>>>(logits,
                                             bias,
                                             weights,
                                             indices,
                                             n_tokens,
                                             topk_group,
                                             topk,
                                             scaling_factor);
}

void launch_grouped_topk_sigmoid_kernel(const float* logits,
                                        const float* bias,
                                        float* weights,
                                        int* indices,
                                        const int64_t n_tokens,
                                        const int n_experts,
                                        const int n_expert_groups,
                                        const int topk_group,
                                        const int topk,
                                        const float scaling_factor,
                                        cudaStream_t stream) {
  static constexpr int WARPS_PER_CTA = 4;

#define LAUNCH_KERNEL(EXPERTS, EXPERTS_GROUPS)                                 \
  launch_grouped_topk_sigmoid_kernel_<EXPERTS, EXPERTS_GROUPS, WARPS_PER_CTA>( \
      logits,                                                                  \
      bias,                                                                    \
      weights,                                                                 \
      indices,                                                                 \
      n_tokens,                                                                \
      topk_group,                                                              \
      topk,                                                                    \
      scaling_factor,                                                          \
      stream);

  switch (n_experts) {
    case 128: {
      switch (n_expert_groups) {
        case 4: {
          LAUNCH_KERNEL(128, 4);
          break;
        }
        case 8: {
          LAUNCH_KERNEL(128, 8);
          break;
        }
        default: {
          CHECK(false) << "Unsupported number of experts: " << n_experts
                       << " and number of expert groups: " << n_expert_groups;
        }
      }
      break;
    }
    case 256: {
      switch (n_expert_groups) {
        case 8: {
          LAUNCH_KERNEL(256, 8);
          break;
        }
        case 16: {
          LAUNCH_KERNEL(256, 16);
          break;
        }
        default: {
          CHECK(false) << "Unsupported number of experts: " << n_experts
                       << " and number of expert groups: " << n_expert_groups;
        }
      }
      break;
    }
    default: {
      CHECK(false) << "Unsupported number of experts: " << n_experts;
    }
  }
}

}  // namespace

void grouped_topk_sigmoid(
    const torch::Tensor& gating_logits,    // [n_tokens, n_experts]
    const torch::Tensor& correction_bias,  // [n_experts]
    const int n_expert_groups,
    const int topk_group,
    const int topk,
    float scaling_factor,
    torch::Tensor& topk_weights,  // [n_tokens, topk]
    torch::Tensor& topk_indices   // [n_tokens, topk]
) {
  const int n_tokens = gating_logits.size(0);
  const int n_experts = gating_logits.size(1);

  CHECK(n_experts % n_expert_groups == 0)
      << "Number of experts must be divisible by number of expert groups, "
         "but got: "
      << n_experts << " and " << n_expert_groups;

  const bool is_pow_2 =
      (n_experts != 0) && ((n_experts & (n_experts - 1)) == 0);
  CHECK(is_pow_2) << "Number of experts must be a power of 2, but got: "
                  << n_experts;

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_grouped_topk_sigmoid_kernel(gating_logits.const_data_ptr<float>(),
                                     correction_bias.const_data_ptr<float>(),
                                     topk_weights.data_ptr<float>(),
                                     topk_indices.data_ptr<int>(),
                                     n_tokens,
                                     n_experts,
                                     n_expert_groups,
                                     topk_group,
                                     topk,
                                     scaling_factor,
                                     stream);
}

}  // namespace llm::kernel