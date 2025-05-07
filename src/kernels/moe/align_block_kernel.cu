#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include <cub/cub.cuh>
#include <cute/config.hpp>
#include <cute/numeric/numeric_types.hpp>
#include <cute/tensor.hpp>

#include "../dispatch.h"
#include "cute/int_tuple.hpp"
#include "cute/layout.hpp"

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
    int n_flatten_tokens,
    int shard_size) {
  // which shard this thread would take care of
  const auto cur_shard = blockIdx.x * blockDim.x + threadIdx.x;
  const auto shard_start = cur_shard * shard_size;
  const auto shard_end = min((cur_shard + 1) * shard_size, n_flatten_tokens);

  for (int i = shard_start; i < shard_end; ++i) {
    const auto e_idx = topk_ids[i];
    // N.B. token ids for each expert is not sorted
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
    size_t n_flatten_tokens,
    scalar_t* __restrict__ cu_sum  // [n_experts+1]
) {
  using namespace cute;
  constexpr int32_t WARP_SIZE = 32;
  // which shard and expert this thread would take care of
  const auto n_shards = blockDim.x;
  const auto cur_shard = threadIdx.x;
  const auto cur_expert = threadIdx.x;

  // number of tokens per shard
  const auto shard_size = ceil_div(n_flatten_tokens, n_shards);
  const auto shard_start = cur_shard * shard_size;
  const auto shard_end = min((cur_shard + 1) * shard_size, n_flatten_tokens);

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

  // process the token shard
  for (int i = shard_start; i < shard_end; ++i) {
    const auto expert_id = topk_ids[i];
    // accumulate token counts for each expert
    atomicAdd(&token_counts[expert_id], 1);
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    cu_sum[0] = 0;
    for (int e_idx = 1; e_idx <= n_experts; ++e_idx) {
      cu_sum[e_idx] = cu_sum[e_idx - 1] +
                      cute::round_up(token_counts[e_idx - 1], block_size);
    }
    *n_padded_permuted_tokens = cu_sum[n_experts];
  }

  __syncthreads();

  // update the expert id for each block
  if (cur_expert < n_experts) {
    for (int i = cu_sum[cur_expert]; i < cu_sum[cur_expert + 1];
         i += block_size) {
      expert_ids[i / block_size] = cur_expert;
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
    int32_t n_flatten_tokens) {
  using namespace cute;

  // which shard and expert this thread would take care of
  const auto n_shards = blockDim.x;
  const auto cur_shard = threadIdx.x;
  const auto cur_expert = threadIdx.x;

  // number of tokens per shard
  const auto shard_size = ceil_div(n_flatten_tokens, n_shards);
  const auto shard_start = cur_shard * shard_size;
  const auto shard_end = min((cur_shard + 1) * shard_size, n_flatten_tokens);

  extern __shared__ int32_t s_mem[];

  // [n_experts+1]
  Tensor cu_sum = make_tensor(make_smem_ptr(reinterpret_cast<int32_t*>(s_mem)),
                              make_layout(make_shape(n_experts + 1)));
  // [n_shards+1][n_experts]
  // token_counts(0, _) = 0 is used to facilitate the prefix sum
  Tensor token_counts = make_tensor(
      make_smem_ptr(reinterpret_cast<int32_t*>(s_mem + n_experts + 1)),
      make_layout(make_shape(n_shards + 1, n_experts)));

  // init token counts for each expert in the shard
  for (int e_idx = 0; e_idx < n_experts; ++e_idx) {
    token_counts(cur_shard + 1, e_idx) = 0;
  }

  // calculate expert counts for each token block
  for (int i = shard_start; i < shard_end; ++i) {
    ++token_counts(cur_shard + 1, topk_ids[i]);
  }

  __syncthreads();

  // calculate the prefix sum for each expert
  // total number of tokens per expert is stored in token_counts(n_shards, _)
  if (cur_expert < n_experts) {
    token_counts(0, cur_expert) = 0;
    for (int i = 1; i <= n_shards; ++i) {
      token_counts(i, cur_expert) += token_counts(i - 1, cur_expert);
    }
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    // caluculate cumulative sum for each expert
    cu_sum[0] = 0;
    for (int e_idx = 1; e_idx <= n_experts; ++e_idx) {
      cu_sum[e_idx] =
          cu_sum[e_idx - 1] +
          cute::round_up(token_counts(n_shards, e_idx - 1), block_size);
    }
    *n_padded_permuted_tokens = cu_sum[n_experts];
  }

  __syncthreads();

  // each thread fills the expert id for each token block
  if (cur_expert < n_experts) {
    for (int i = cu_sum[cur_expert]; i < cu_sum[cur_expert + 1];
         i += block_size) {
      expert_ids[i / block_size] = cur_expert;
    }
  }

  for (int i = shard_start; i < shard_end; ++i) {
    const auto e_idx = topk_ids[i];
    const auto idx_in_shard = token_counts(cur_shard, e_idx)++;
    sorted_token_ids[cu_sum[e_idx] + idx_in_shard] = i;
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
  const auto n_flatten_tokens = topk_ids.numel();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  DISPATCH_INTEGRAL_TYPES(topk_ids.scalar_type(), "align_block_kernel", [&] {
    if (n_flatten_tokens <= 1024 && n_experts <= 64) {
      const int32_t n_shards = max((int32_t)n_experts, WARP_SIZE);
      const int32_t smem_size =
          ((n_shards + 1) * n_experts + (n_experts + 1)) * sizeof(int32_t);

      align_block_kernel<scalar_t><<<1, n_shards, smem_size, stream>>>(
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
      int64_t padded_num_experts = cute::round_up(n_experts, experts_per_warp);
      size_t num_warps = cute::ceil_div(padded_num_experts, experts_per_warp);
      size_t shared_mem_size = num_warps * experts_per_warp * sizeof(int32_t);

      cusum_kernel<scalar_t><<<1, threads, shared_mem_size, stream>>>(
          topk_ids.data_ptr<scalar_t>(),
          experts_ids.data_ptr<int32_t>(),
          n_padded_permuted_tokens.data_ptr<int32_t>(),
          n_experts,
          padded_num_experts,
          experts_per_warp,
          block_size,
          n_flatten_tokens,
          cu_sum.data_ptr<int32_t>());

      // use up to 256 threads to sort
      const int threads = 256;
      // partition permuted tokens into blocks
      int n_blocks = cute::ceil_div(n_flatten_tokens, threads);
      // up to 65535 blocks
      n_blocks = std::min(n_blocks, 65535);
      const auto shard_size =
          cute::ceil_div(n_flatten_tokens, n_blocks * threads);
      row_id_map_kernel<scalar_t><<<n_blocks, threads, 0, stream>>>(
          topk_ids.data_ptr<scalar_t>(),
          sorted_token_ids.data_ptr<scalar_t>(),
          cu_sum.data_ptr<scalar_t>(),
          n_flatten_tokens,
          shard_size);
    }
  });
}

}  // namespace llm::kernel::moe
