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
template <typename T>
inline T* data_ptr(torch::Tensor& t) {
  return reinterpret_cast<T*>(t.data_ptr());
}

template <typename T>
inline const T* const_data_ptr(torch::Tensor& t) {
  return reinterpret_cast<const T*>(t.const_data_ptr());
}

template <typename scalar_t>
__global__ void count_and_sort_expert_tokens_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ cumsum_buffer,
    size_t numel) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;

  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i];
    int32_t rank_post_pad = atomicAdd(&cumsum_buffer[expert_id], 1);
    sorted_token_ids[rank_post_pad] = i;
  }
}

template <typename scalar_t>
__global__ void moe_align_block_size_kernel(
    const scalar_t* __restrict__ topk_ids,        // [n_tokens, topk]
    int32_t* __restrict__ sorted_token_ids,       // [n_permuted_tokens+]
    int32_t* __restrict__ expert_ids,             // [n_blocks+]
    int32_t* __restrict__ total_tokens_post_pad,  // [1]
    int32_t num_experts,
    int32_t padded_num_experts,
    int32_t experts_per_warp,
    int32_t block_size,
    size_t numel,
    int32_t* __restrict__ cumsum  // [n_experts+1]
) {
  constexpr int32_t WARP_SIZE = 32;
  // [n_experts+1]
  extern __shared__ int32_t shared_counts[];

  const int warp_id = threadIdx.x / WARP_SIZE;
  const int my_expert_start = warp_id * experts_per_warp;

  // init token counts for each thread
  for (int i = 0; i < experts_per_warp; ++i) {
    if (my_expert_start + i < padded_num_experts) {
      shared_counts[warp_id * experts_per_warp + i] = 0;
    }
  }

  __syncthreads();

  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;

  // process the token shard
  for (size_t i = tid; i < numel; i += stride) {
    int expert_id = topk_ids[i];
    int warp_idx = expert_id / experts_per_warp;
    int expert_offset = expert_id % experts_per_warp;
    // accumulate token counts for each expert
    atomicAdd(&shared_counts[warp_idx * experts_per_warp + expert_offset], 1);
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      int expert_count = 0;
      int warp_idx = (i - 1) / experts_per_warp;
      int expert_offset = (i - 1) % experts_per_warp;
      expert_count = shared_counts[warp_idx * experts_per_warp + expert_offset];
      // why not just expert_count = shared_counts[i - 1]?

      cumsum[i] = cumsum[i - 1] + cute::round_up(expert_count, block_size);
    }
    *total_tokens_post_pad = cumsum[num_experts];
  }

  __syncthreads();

  // update the expert id for each block
  if (threadIdx.x < num_experts) {
    for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1];
         i += block_size) {
      expert_ids[i / block_size] = threadIdx.x;
    }
  }
}

template <typename scalar_t>
__global__ void small_align_block_kernel(
    const scalar_t* __restrict__ topk_ids,        // [n_tokens, topk]
    int32_t* __restrict__ sorted_token_ids,       // [n_permuted_tokens+]
    int32_t* __restrict__ expert_ids,             // [n_blocks+]
    int32_t* __restrict__ total_tokens_post_pad,  // [1]
    int32_t num_experts,
    int32_t block_size,
    size_t numel) {
  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;

  //
  extern __shared__ int32_t shared_mem[];
  // [n_experts+1]
  int32_t* cumsum = shared_mem;
  // [n_shards+1][n_experts]
  int32_t* tokens_cnts = (int32_t*)(shared_mem + num_experts + 1);

  // init token counts for each expert in the shard
  for (int i = 0; i < num_experts; ++i) {
    tokens_cnts[(threadIdx.x + 1) * num_experts + i] = 0;
  }

  // calculate expert counts for each token block
  for (size_t i = tid; i < numel; i += stride) {
    // ++tokens_cnts[threadIdx.x+1][topk_ids[i]];
    ++tokens_cnts[(threadIdx.x + 1) * num_experts + topk_ids[i]];
  }

  __syncthreads();

  // calculate the prefix sum of token counts for each expert within the block
  if (threadIdx.x < num_experts) {
    tokens_cnts[threadIdx.x] = 0;
    for (int i = 1; i <= blockDim.x; ++i) {
      tokens_cnts[i * num_experts + threadIdx.x] +=
          tokens_cnts[(i - 1) * num_experts + threadIdx.x];
    }
  }

  __syncthreads();

  // caluculate token counts for each expert
  if (threadIdx.x == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      cumsum[i] = cumsum[i - 1] +
                  cute::round_up(tokens_cnts[blockDim.x * num_experts + i - 1],
                                 block_size);
    }
    *total_tokens_post_pad = static_cast<int32_t>(cumsum[num_experts]);
  }

  __syncthreads();

  // each thread fills the expert id for each token
  if (threadIdx.x < num_experts) {
    for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1];
         i += block_size) {
      expert_ids[i / block_size] = threadIdx.x;
    }
  }

  // each thread process one block
  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i];
    int32_t rank_post_pad =
        tokens_cnts[threadIdx.x * num_experts + expert_id] + cumsum[expert_id];
    sorted_token_ids[rank_post_pad] = i;
    ++tokens_cnts[threadIdx.x * num_experts + expert_id];
  }
}

}  // namespace

void permute_align_block(torch::Tensor topk_ids,
                         int64_t num_experts,
                         int64_t block_size,
                         torch::Tensor sorted_token_ids,
                         torch::Tensor experts_ids,
                         torch::Tensor num_tokens_post_pad,
                         torch::Tensor cumsum_buffer) {
  constexpr int threads = 1024;
  constexpr int32_t WARP_SIZE = 32;

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  DISPATCH_INTEGRAL_TYPES(topk_ids.scalar_type(), "align_block_kernel", [&] {
    bool small_batch_expert_mode =
        (topk_ids.numel() < 1024) && (num_experts <= 64);

    if (small_batch_expert_mode) {
      const int32_t threads = max((int32_t)num_experts, WARP_SIZE);
      const int32_t shared_mem_size =
          ((threads + 1) * num_experts + (num_experts + 1)) * sizeof(int32_t);

      auto small_batch_expert_kernel = small_align_block_kernel<scalar_t>;
      small_batch_expert_kernel<<<1, threads, shared_mem_size, stream>>>(
          topk_ids.data_ptr<scalar_t>(),
          sorted_token_ids.data_ptr<int32_t>(),
          experts_ids.data_ptr<int32_t>(),
          num_tokens_post_pad.data_ptr<int32_t>(),
          num_experts,
          block_size,
          topk_ids.numel());
    } else {
      // why it is faster?
      // use more sms to sort
      auto align_kernel = moe_align_block_size_kernel<scalar_t>;

      int experts_per_warp = WARP_SIZE;
      int64_t padded_num_experts =
          ((num_experts + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
      size_t num_warps = cute::ceil_div(padded_num_experts, experts_per_warp);
      size_t shared_mem_size = num_warps * experts_per_warp * sizeof(int32_t);

      // can be removed.
      // [n_experts+1]
      cumsum_buffer.zero_();
      // threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

      align_kernel<<<1, threads, shared_mem_size, stream>>>(
          topk_ids.data_ptr<scalar_t>(),
          sorted_token_ids.data_ptr<int32_t>(),
          experts_ids.data_ptr<int32_t>(),
          num_tokens_post_pad.data_ptr<int32_t>(),
          num_experts,
          padded_num_experts,
          experts_per_warp,
          block_size,
          topk_ids.numel(),
          cumsum_buffer.data_ptr<int32_t>());

      // use up to 256 threads to sort
      const int block_threads = std::min(256, (int)threads);
      // partition permuted tokens into blocks
      const int num_blocks =
          (topk_ids.numel() + block_threads - 1) / block_threads;
      const int max_blocks = 65535;
      const int actual_blocks = std::min(num_blocks, max_blocks);

      auto sort_kernel = count_and_sort_expert_tokens_kernel<scalar_t>;
      sort_kernel<<<actual_blocks, block_threads, 0, stream>>>(
          topk_ids.data_ptr<scalar_t>(),
          sorted_token_ids.data_ptr<int32_t>(),
          cumsum_buffer.data_ptr<int32_t>(),
          topk_ids.numel());
    }
  });
}

}  // namespace llm::kernel::moe
