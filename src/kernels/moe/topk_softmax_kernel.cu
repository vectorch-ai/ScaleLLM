// adapted from https://github.com/NVIDIA/FasterTransformer
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cutlass/array.h>
#include <torch/torch.h>

#include <algorithm>
#include <cfloat>

// #include "../dispatch.h"
// #include "../reduce_kernel_utils.cuh"

// Ignore CUTLASS warnings about type punning
#ifdef __GNUC__  // Check if the compiler is GCC or Clang
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

namespace llm::kernel {
static constexpr int WARP_SIZE = 32;

// ====================== TopK softmax things ===============================

/*
  A Top-K gating softmax written to exploit when the number of experts in the
  MoE layers are a small power of 2. This allows us to cleanly share the rows
  among the threads in a single warp and eliminate communication between warps
  (so no need to use shared mem).

  It fuses the softmax, max and argmax into a single kernel.

  Limitations:
  1) This implementation is intended for when the number of experts is a small
  power of 2. 2) This implementation assumes k is small, but will work for any
  k.
*/

template <int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE) __global__
    void topkGatingSoftmax(float const* input,
                           bool const* finished,
                           float* output,
                           int64_t const num_rows,
                           int* indices,
                           int const k,
                           int const start_expert,
                           int const end_expert) {
  // We begin by enforcing compile time assertions and setting up compile time
  // constants.
  static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
  static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS),
                "NUM_EXPERTS must be power of 2");
  static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG),
                "BYTES_PER_LDG must be power of 2");
  static_assert(BYTES_PER_LDG <= 16, "BYTES_PER_LDG must be leq 16");

  // Number of bytes each thread pulls in per load
  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(float);
  static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
  static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
  static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

  // Restrictions based on previous section.
  static_assert(
      VPT % ELTS_PER_LDG == 0,
      "The elements per thread must be a multiple of the elements per ldg");
  static_assert(WARP_SIZE % THREADS_PER_ROW == 0,
                "The threads per row must cleanly divide the threads per warp");
  static_assert(THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW),
                "THREADS_PER_ROW must be power of 2");
  static_assert(THREADS_PER_ROW <= WARP_SIZE,
                "THREADS_PER_ROW can be at most warp size");

  // We have NUM_EXPERTS elements per row. We specialize for small #experts
  static constexpr int ELTS_PER_WARP = WARP_SIZE * VPT;
  static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
  static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

  // Restrictions for previous section.
  static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0,
                "The elts per row must cleanly divide the total elt per warp");

  // ===================== From this point, we finally start computing run-time
  // variables. ========================

  // Compute CTA and warp rows. We pack multiple rows into a single warp, and a
  // block contains WARPS_PER_CTA warps. This, each block processes a chunk of
  // rows. We start by computing the start row for each block.
  int64_t const cta_base_row = blockIdx.x * ROWS_PER_CTA;

  // Now, using the base row per thread block, we compute the base row per warp.
  int64_t const warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;

  // The threads in a warp are split into sub-groups that will work on a row.
  // We compute row offset for each thread sub-group
  int const thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
  int64_t const thread_row = warp_base_row + thread_row_in_warp;

  // Threads with indices out of bounds should early exit here.
  if (thread_row >= num_rows) {
    return;
  }
  bool const row_is_active = finished ? !finished[thread_row] : true;

  // We finally start setting up the read pointers for each thread. First, each
  // thread jumps to the start of the row it will read.
  float const* thread_row_ptr = input + thread_row * ELTS_PER_ROW;

  // Now, we compute the group each thread belong to in order to determine the
  // first column to start loads.
  int const thread_group_idx = threadIdx.x % THREADS_PER_ROW;
  int const first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
  float const* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

  // Determine the pointer type to use to read in the data depending on the
  // BYTES_PER_LDG template param. In theory, this can support all powers of 2
  // up to 16.
  using AccessType = cutlass::AlignedArray<float, ELTS_PER_LDG>;

  // Finally, we pull in the data from global mem
  cutlass::Array<float, VPT> row_chunk;
  AccessType* row_chunk_vec_ptr = reinterpret_cast<AccessType*>(&row_chunk);
  AccessType const* vec_thread_read_ptr =
      reinterpret_cast<AccessType const*>(thread_read_ptr);
#pragma unroll
  for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
    row_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
  }

  // First, we perform a max reduce within the thread. We can do the max in fp16
  // safely (I think) and just convert to float afterwards for the exp + sum
  // reduction.
  float thread_max = row_chunk[0];
#pragma unroll
  for (int ii = 1; ii < VPT; ++ii) {
    thread_max = max(thread_max, row_chunk[ii]);
  }

// Now, we find the max within the thread group and distribute among the
// threads. We use a butterfly reduce.
#pragma unroll
  for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
    thread_max =
        max(thread_max,
            __shfl_xor_sync(0xFFFFFFFF, thread_max, mask, THREADS_PER_ROW));
  }

  // From this point, thread max in all the threads have the max within the row.
  // Now, we subtract the max from each element in the thread and take the exp.
  // We also compute the thread local sum.
  float row_sum = 0;
#pragma unroll
  for (int ii = 0; ii < VPT; ++ii) {
    row_chunk[ii] = expf(row_chunk[ii] - thread_max);
    row_sum += row_chunk[ii];
  }

// Now, we perform the sum reduce within each thread group. Similar to the max
// reduce, we use a bufferfly pattern.
#pragma unroll
  for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
    row_sum += __shfl_xor_sync(0xFFFFFFFF, row_sum, mask, THREADS_PER_ROW);
  }

  // From this point, all threads have the max and the sum for their rows in the
  // thread_max and thread_sum variables respectively. Finally, we can scale the
  // rows for the softmax. Technically, for top-k gating we don't need to
  // compute the entire softmax row. We can likely look at the maxes and only
  // compute for the top-k values in the row. However, this kernel will likely
  // not be a bottle neck and it seems better to closer match torch and find the
  // argmax after computing the softmax.
  float const reciprocal_row_sum = 1.f / row_sum;

#pragma unroll
  for (int ii = 0; ii < VPT; ++ii) {
    row_chunk[ii] = row_chunk[ii] * reciprocal_row_sum;
  }

  // Now, softmax_res contains the softmax of the row chunk. Now, I want to find
  // the topk elements in each row, along with the max index.
  int start_col = first_elt_read_by_thread;
  static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;

  for (int k_idx = 0; k_idx < k; ++k_idx) {
    // First, each thread does the local argmax
    float max_val = row_chunk[0];
    int expert = start_col;
#pragma unroll
    for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD;
         ++ldg, col += COLS_PER_GROUP_LDG) {
#pragma unroll
      for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
        float val = row_chunk[ldg * ELTS_PER_LDG + ii];

        // No check on the experts here since columns with the smallest index
        // are processed first and only updated if > (not >=)
        if (val > max_val) {
          max_val = val;
          expert = col + ii;
        }
      }
    }

// Now, we perform the argmax reduce. We use the butterfly pattern so threads
// reach consensus about the max. This will be useful for K > 1 so that the
// threads can agree on "who" had the max value. That thread can then blank out
// their max with -inf and the warp can run more iterations...
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      float other_max =
          __shfl_xor_sync(0xFFFFFFFF, max_val, mask, THREADS_PER_ROW);
      int other_expert =
          __shfl_xor_sync(0xFFFFFFFF, expert, mask, THREADS_PER_ROW);

      // We want lower indices to "win" in every thread so we break ties this
      // way
      if (other_max > max_val ||
          (other_max == max_val && other_expert < expert)) {
        max_val = other_max;
        expert = other_expert;
      }
    }

    // Write the max for this k iteration to global memory.
    if (thread_group_idx == 0) {
      // Add a guard to ignore experts not included by this node
      bool const node_uses_expert =
          expert >= start_expert && expert < end_expert;
      bool const should_process_row = row_is_active && node_uses_expert;

      // The lead thread from each sub-group will write out the final results to
      // global memory. (This will be a single) thread per row of the
      // input/output matrices.
      int64_t const idx = k * thread_row + k_idx;
      output[idx] = max_val;
      indices[idx] =
          should_process_row ? (expert - start_expert) : (NUM_EXPERTS + expert);
    }

    // Finally, we clear the value in the thread with the current max if there
    // is another iteration to run.
    if (k_idx + 1 < k) {
      int const ldg_group_for_expert = expert / COLS_PER_GROUP_LDG;
      int const thread_to_clear_in_group =
          (expert / ELTS_PER_LDG) % THREADS_PER_ROW;

      // Only the thread in the group which produced the max will reset the
      // "winning" value to -inf.
      if (thread_group_idx == thread_to_clear_in_group) {
        int const offset_for_expert = expert % ELTS_PER_LDG;
        // Safe to set to any negative value since row_chunk values must be
        // between 0 and 1.
        row_chunk[ldg_group_for_expert * ELTS_PER_LDG + offset_for_expert] =
            -10000.f;
      }
    }
  }
}

namespace detail {
// Constructs some constants needed to partition the work across threads at
// compile time.
template <int EXPERTS, int BYTES_PER_LDG>
struct TopkConstants {
  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(float);
  static_assert(EXPERTS / (ELTS_PER_LDG * WARP_SIZE) == 0 ||
                EXPERTS % (ELTS_PER_LDG * WARP_SIZE) == 0);
  static constexpr int VECs_PER_THREAD =
      std::max(1, EXPERTS / (ELTS_PER_LDG * WARP_SIZE));
  static constexpr int VPT = VECs_PER_THREAD * ELTS_PER_LDG;
  static constexpr int THREADS_PER_ROW = EXPERTS / VPT;
  static constexpr int ROWS_PER_WARP = WARP_SIZE / THREADS_PER_ROW;
};
}  // namespace detail

template <int EXPERTS, int WARPS_PER_TB>
void topkGatingSoftmaxLauncherHelper(float const* input,
                                     bool const* finished,
                                     float* output,
                                     int* indices,
                                     int64_t const num_rows,
                                     int const k,
                                     int const start_expert,
                                     int const end_expert,
                                     cudaStream_t stream) {
  static constexpr std::size_t MAX_BYTES_PER_LDG = 16;

  static constexpr int BYTES_PER_LDG =
      std::min(MAX_BYTES_PER_LDG, sizeof(float) * EXPERTS);
  using Constants = detail::TopkConstants<EXPERTS, BYTES_PER_LDG>;
  static constexpr int VPT = Constants::VPT;
  static constexpr int ROWS_PER_WARP = Constants::ROWS_PER_WARP;
  int64_t const num_warps = (num_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
  int64_t const num_blocks = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB;

  dim3 block_dim(WARP_SIZE, WARPS_PER_TB);
  topkGatingSoftmax<VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG>
      <<<num_blocks, block_dim, 0, stream>>>(input,
                                             finished,
                                             output,
                                             num_rows,
                                             indices,
                                             k,
                                             start_expert,
                                             end_expert);
}

void topkGatingSoftmaxKernelLauncher(float const* input,
                                     float* output,
                                     int* indices,
                                     int64_t const num_rows,
                                     int const num_experts,
                                     int const k,
                                     cudaStream_t stream) {
  static constexpr int WARPS_PER_TB = 4;
  int const start_expert = 0;
  int const end_expert = num_experts;

  switch (num_experts) {
    case 1: {
      topkGatingSoftmaxLauncherHelper<1, WARPS_PER_TB>(input,
                                                       nullptr,
                                                       output,
                                                       indices,
                                                       num_rows,
                                                       k,
                                                       start_expert,
                                                       end_expert,
                                                       stream);
      break;
    }
    case 2: {
      topkGatingSoftmaxLauncherHelper<2, WARPS_PER_TB>(input,
                                                       nullptr,
                                                       output,
                                                       indices,
                                                       num_rows,
                                                       k,
                                                       start_expert,
                                                       end_expert,
                                                       stream);
      break;
    }
    case 4: {
      topkGatingSoftmaxLauncherHelper<4, WARPS_PER_TB>(input,
                                                       nullptr,
                                                       output,
                                                       indices,
                                                       num_rows,
                                                       k,
                                                       start_expert,
                                                       end_expert,
                                                       stream);
      break;
    }
    case 8: {
      topkGatingSoftmaxLauncherHelper<8, WARPS_PER_TB>(input,
                                                       nullptr,
                                                       output,
                                                       indices,
                                                       num_rows,
                                                       k,
                                                       start_expert,
                                                       end_expert,
                                                       stream);
      break;
    }
    case 16: {
      topkGatingSoftmaxLauncherHelper<16, WARPS_PER_TB>(input,
                                                        nullptr,
                                                        output,
                                                        indices,
                                                        num_rows,
                                                        k,
                                                        start_expert,
                                                        end_expert,
                                                        stream);
      break;
    }
    case 32: {
      topkGatingSoftmaxLauncherHelper<32, WARPS_PER_TB>(input,
                                                        nullptr,
                                                        output,
                                                        indices,
                                                        num_rows,
                                                        k,
                                                        start_expert,
                                                        end_expert,
                                                        stream);
      break;
    }
    case 64: {
      topkGatingSoftmaxLauncherHelper<64, WARPS_PER_TB>(input,
                                                        nullptr,
                                                        output,
                                                        indices,
                                                        num_rows,
                                                        k,
                                                        start_expert,
                                                        end_expert,
                                                        stream);
      break;
    }
    case 128: {
      topkGatingSoftmaxLauncherHelper<128, WARPS_PER_TB>(input,
                                                         nullptr,
                                                         output,
                                                         indices,
                                                         num_rows,
                                                         k,
                                                         start_expert,
                                                         end_expert,
                                                         stream);
      break;
    }
    case 256: {
      topkGatingSoftmaxLauncherHelper<256, WARPS_PER_TB>(input,
                                                         nullptr,
                                                         output,
                                                         indices,
                                                         num_rows,
                                                         k,
                                                         start_expert,
                                                         end_expert,
                                                         stream);
      break;
    }
    default: {
      CHECK(false) << "Unsupported number of experts: " << num_experts;
    }
  }
}

void topk_softmax(const torch::Tensor& gating_logit,  // [n_tokens, n_experts]
                  torch::Tensor& topk_weights,        // [n_tokens, topk]
                  torch::Tensor& topk_indices         // [n_tokens, topk]
) {
  const int n_tokens = gating_logit.size(0);
  const int n_experts = gating_logit.size(1);
  const int topk = topk_weights.size(-1);

  // const bool is_pow_2 =
  //     (num_experts != 0) && ((num_experts & (num_experts - 1)) == 0);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  topkGatingSoftmaxKernelLauncher(gating_logit.data_ptr<float>(),
                                  topk_weights.data_ptr<float>(),
                                  topk_indices.data_ptr<int>(),
                                  n_tokens,
                                  n_experts,
                                  topk,
                                  stream);
}

}  // namespace llm::kernel