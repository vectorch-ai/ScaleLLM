#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include <cub/cub.cuh>
#include <cute/config.hpp>
#include <cute/numeric/numeric_types.hpp>
#include <cute/tensor.hpp>

#include "../dispatch.h"
#include "cute/int_tuple.hpp"

// Adapated from
// https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/moe/moe_align_kernel.cu

// clang-format off
// for exmple: n_tokens = 2, n_experts = 8, topk = 2
// f_idx: idx in flatten indices
// p_idx: idx in permuted tokens
// k_idx: topk idx
// t_idx: token idx
// row_id_map: [topk, n_tokens] => idx in permuted tokens
//  ______________________________________________________________________________________
// |                 |     flatten indices         |           sort indices               |
// |    Steps        |   sort by (tokens, topk)    |        by (experts, tokens)          |
// |_________________|_____________________________|______________________________________|
// |                 |    [n_tokens * topk]        |     [n_tokens * topk] => f_idx       |
// |     Dim         |                             |   f_idx: idx in flatten indices      |
// |_________________|_____________________________|______________________________________|
// |                 |                             |                                      |
// |      top0, top1 |   f_idx: | 0 | 1 | 2 | 3 |  |   p_idx: |  0  |  1  |  2  |  3  |   |
// | t0 -> [e2, e1]  | experts: | 2 | 1 | 2 | 5 |  |   f_idx: |  1  |  0  |  2  |  3  |   |
// | t1 -> [e2, e5]  |  tokens: |   t0  |   t1  |  |  tokens: |  t0 |  t0 |  t1 |  t1 |   |
// |                 |                             | experts: |  e1 |     e2    |  e5 |   |
// |                 |                             |                                      |
// |                 |                             |                                      |
// |_________________|_____________________________|______________________________________|
// clang-format on

namespace llm::kernel::moe {

namespace {
// map p_idx to f_idx
template <typename scalar_t>
__global__ void row_id_map_kernel(
    const scalar_t* __restrict__ topk_ids,    // [n_tokens, topk]
    scalar_t* __restrict__ sorted_token_ids,  // [n_permuted_tokens+]
    scalar_t* __restrict__ cu_sum,            // [n_experts+1]
    int n_permuted_tokens) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  for (int i = tid; i < n_permuted_tokens; i += stride) {
    const auto e_idx = topk_ids[i];
    const auto p_idx = atomicAdd(&cu_sum[e_idx], 1);
    sorted_token_ids[p_idx] = i;
  }
}

template <typename scalar_t>
__global__ void cusum_kernel(
    const scalar_t* __restrict__ topk_ids,            // [n_tokens, topk]
    scalar_t* __restrict__ expert_ids,                // [n_blocks+]
    scalar_t* __restrict__ n_padded_permuted_tokens,  // [1]
    int32_t n_experts,
    int32_t n_padded_experts,
    int32_t experts_per_warp,
    int32_t block_size,
    size_t n_permuted_tokens,
    scalar_t* __restrict__ cu_sum  // [n_experts+1]
) {
  constexpr int32_t WARP_SIZE = 32;
  // [n_experts+1]
  extern __shared__ int32_t token_counts[];

  const int warp_id = threadIdx.x / WARP_SIZE;
  const int my_expert_start = warp_id * experts_per_warp;

  // init token counts for each thread
  for (int i = 0; i < experts_per_warp; ++i) {
    if (my_expert_start + i < n_padded_experts) {
      token_counts[warp_id * experts_per_warp + i] = 0;
    }
  }

  __syncthreads();

  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;

  // process the token shard
  for (size_t i = tid; i < n_permuted_tokens; i += stride) {
    int expert_id = topk_ids[i];
    int warp_idx = expert_id / experts_per_warp;
    int expert_offset = expert_id % experts_per_warp;
    // accumulate token counts for each expert
    atomicAdd(&token_counts[warp_idx * experts_per_warp + expert_offset], 1);
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    cu_sum[0] = 0;
    for (int i = 1; i <= n_experts; ++i) {
      int expert_count = 0;
      int warp_idx = (i - 1) / experts_per_warp;
      int expert_offset = (i - 1) % experts_per_warp;
      expert_count = token_counts[warp_idx * experts_per_warp + expert_offset];
      // why not just expert_count = shared_counts[i - 1]?

      cu_sum[i] = cu_sum[i - 1] + cute::round_up(expert_count, block_size);
    }
    *n_padded_permuted_tokens = cu_sum[n_experts];
  }

  __syncthreads();

  // update the expert id for each block
  if (threadIdx.x < n_experts) {
    for (int i = cu_sum[threadIdx.x]; i < cu_sum[threadIdx.x + 1];
         i += block_size) {
      expert_ids[i / block_size] = threadIdx.x;
    }
  }
}

template <typename scalar_t>
__global__ void align_block_kernel(
    const scalar_t* __restrict__ topk_ids,           // [n_tokens, topk]
    int32_t* __restrict__ sorted_token_ids,          // [n_permuted_tokens+]
    int32_t* __restrict__ expert_ids,                // [n_blocks+]
    int32_t* __restrict__ n_padded_permuted_tokens,  // [1]
    int32_t n_experts,
    int32_t block_size,
    size_t n_permuted_tokens) {
  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;

  //
  extern __shared__ int32_t s_mem[];
  // [n_experts+1]
  int32_t* cu_sum = s_mem;
  // [n_shards+1][n_experts]
  int32_t* token_counts = (int32_t*)(s_mem + n_experts + 1);

  // init token counts for each expert in the shard
  for (int i = 0; i < n_experts; ++i) {
    token_counts[(threadIdx.x + 1) * n_experts + i] = 0;
  }

  // calculate expert counts for each token block
  for (size_t i = tid; i < n_permuted_tokens; i += stride) {
    // ++tokens_cnts[threadIdx.x+1][topk_ids[i]];
    ++token_counts[(threadIdx.x + 1) * n_experts + topk_ids[i]];
  }

  __syncthreads();

  // calculate the prefix sum of token counts for each expert within the block
  if (threadIdx.x < n_experts) {
    token_counts[threadIdx.x] = 0;
    for (int i = 1; i <= blockDim.x; ++i) {
      token_counts[i * n_experts + threadIdx.x] +=
          token_counts[(i - 1) * n_experts + threadIdx.x];
    }
  }

  __syncthreads();

  // caluculate token counts for each expert
  if (threadIdx.x == 0) {
    cu_sum[0] = 0;
    for (int i = 1; i <= n_experts; ++i) {
      cu_sum[i] = cu_sum[i - 1] +
                  cute::round_up(token_counts[blockDim.x * n_experts + i - 1],
                                 block_size);
    }
    *n_padded_permuted_tokens = cu_sum[n_experts];
  }

  __syncthreads();

  // each thread fills the expert id for each token
  if (threadIdx.x < n_experts) {
    for (int i = cu_sum[threadIdx.x]; i < cu_sum[threadIdx.x + 1];
         i += block_size) {
      expert_ids[i / block_size] = threadIdx.x;
    }
  }

  // each thread process one block
  for (size_t i = tid; i < n_permuted_tokens; i += stride) {
    int32_t expert_id = topk_ids[i];
    int32_t rank_post_pad =
        token_counts[threadIdx.x * n_experts + expert_id] + cu_sum[expert_id];
    sorted_token_ids[rank_post_pad] = i;
    ++token_counts[threadIdx.x * n_experts + expert_id];
  }
}

}  // namespace

void permute_align_block(
    torch::Tensor topk_ids,  // [n_tokens, topk]
    int64_t n_experts,
    int64_t block_size,
    torch::Tensor sorted_token_ids,          // [n_padded_permuted_tokens+]
    torch::Tensor experts_ids,               // [n_blocks+]
    torch::Tensor n_padded_permuted_tokens,  // [1]
    torch::Tensor cu_sum                     // [n_experts+1]
) {
  constexpr int threads = 1024;
  constexpr int32_t WARP_SIZE = 32;

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  DISPATCH_INTEGRAL_TYPES(topk_ids.scalar_type(), "align_block_kernel", [&] {
    // bool small_batch_expert_mode =
    //     (topk_ids.numel() < 1024) && (n_experts <= 64);
    bool small_batch_expert_mode = false;

    if (small_batch_expert_mode) {
      const int32_t threads = max((int32_t)n_experts, WARP_SIZE);
      const int32_t shared_mem_size =
          ((threads + 1) * n_experts + (n_experts + 1)) * sizeof(int32_t);

      align_block_kernel<scalar_t><<<1, threads, shared_mem_size, stream>>>(
          topk_ids.data_ptr<scalar_t>(),
          sorted_token_ids.data_ptr<int32_t>(),
          experts_ids.data_ptr<int32_t>(),
          n_padded_permuted_tokens.data_ptr<int32_t>(),
          n_experts,
          block_size,
          topk_ids.numel());
    } else {
      // why it is faster?
      // use more sms to sort
      int experts_per_warp = WARP_SIZE;
      int64_t padded_num_experts =
          ((n_experts + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
      size_t num_warps = cute::ceil_div(padded_num_experts, experts_per_warp);
      size_t shared_mem_size = num_warps * experts_per_warp * sizeof(int32_t);

      // can be removed.
      // [n_experts+1]
      cu_sum.zero_();
      // threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

      cusum_kernel<scalar_t><<<1, threads, shared_mem_size, stream>>>(
          topk_ids.data_ptr<scalar_t>(),
          experts_ids.data_ptr<int32_t>(),
          n_padded_permuted_tokens.data_ptr<int32_t>(),
          n_experts,
          padded_num_experts,
          experts_per_warp,
          block_size,
          topk_ids.numel(),
          cu_sum.data_ptr<int32_t>());

      // use up to 256 threads to sort
      const int block_threads = std::min(256, (int)threads);
      // partition permuted tokens into blocks
      const int num_blocks =
          (topk_ids.numel() + block_threads - 1) / block_threads;
      const int max_blocks = 65535;
      const int actual_blocks = std::min(num_blocks, max_blocks);

      row_id_map_kernel<scalar_t><<<actual_blocks, block_threads, 0, stream>>>(
          topk_ids.data_ptr<scalar_t>(),
          sorted_token_ids.data_ptr<scalar_t>(),
          cu_sum.data_ptr<scalar_t>(),
          topk_ids.numel());
    }
  });
}

}  // namespace llm::kernel::moe
