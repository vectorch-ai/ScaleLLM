#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include <cute/tensor.hpp>

#include "../dispatch.h"

// Adapated from
// https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/moe/moe_align_kernel.cu

namespace llm::kernel::moe {

namespace {
constexpr int32_t WARP_SIZE = 32;

// map p_idx to f_idx
template <typename scalar_t>
__global__ void row_id_map_kernel(
    const scalar_t* __restrict__ topk_ids,      // [m, topk]
    scalar_t* __restrict__ sorted_token_idxes,  // [n_padded_tokens+]
    scalar_t* __restrict__ cu_sum,              // [n_experts+1]
    int n_tokens                                // m * topk
) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  for (int i = tid; i < n_tokens; i += stride) {
    const auto e_idx = topk_ids[i];
    // N.B. token ids for each expert is not sorted
    const auto p_idx = atomicAdd(&cu_sum[e_idx], 1);
    sorted_token_idxes[p_idx] = i;
  }
}

template <typename scalar_t>
__global__ void cusum_kernel(
    const scalar_t* __restrict__ topk_ids,   // [n_tokens, topk]
    scalar_t* __restrict__ expert_ids,       // [n_blocks+]
    scalar_t* __restrict__ n_padded_tokens,  // [1]
    int n_experts,
    int block_size,
    size_t n_tokens,               // n_tokens * topk
    scalar_t* __restrict__ cu_sum  // [n_experts+1]
) {
  using namespace cute;

  const int tid = threadIdx.x;
  const int stride = blockDim.x;

  const auto curr_expert = threadIdx.x;

  // token count for each expert [n_padded_experts]
  extern __shared__ int token_counts[];

  // init token counts for each expert
  if (curr_expert < n_experts) {
    token_counts[curr_expert] = 0;
  }

  __syncthreads();

  // process the token shard
  for (int i = tid; i < n_tokens; i += stride) {
    // accumulate token counts for each expert
    atomicAdd(&token_counts[topk_ids[i]], 1);
  }

  __syncthreads();

  if (tid == 0) {
    cu_sum[0] = 0;
    for (int e_idx = 1; e_idx <= n_experts; ++e_idx) {
      cu_sum[e_idx] = cu_sum[e_idx - 1] +
                      cute::round_up(token_counts[e_idx - 1], block_size);
    }
    *n_padded_tokens = cu_sum[n_experts];
  }

  __syncthreads();

  // update the expert id for each block
  if (curr_expert < n_experts) {
    for (int i = cu_sum[curr_expert]; i < cu_sum[curr_expert + 1];
         i += block_size) {
      expert_ids[i / block_size] = curr_expert;
    }
  }
}

template <typename scalar_t>
__global__ void align_block_kernel(
    const scalar_t* __restrict__ topk_ids,      // [m, topk]
    scalar_t* __restrict__ sorted_token_idxes,  // [n_padded_tokens+]
    scalar_t* __restrict__ expert_ids,          // [n_blocks+]
    scalar_t* __restrict__ n_padded_tokens,     // [1]
    int n_experts,
    int block_size,
    int n_tokens  // m * topk
) {
  using namespace cute;

  // which shard and expert this thread would take care of
  const int n_shards = blockDim.x;
  const int curr_shard = threadIdx.x;
  const int curr_expert = threadIdx.x;

  const int tid = threadIdx.x;
  const int stride = blockDim.x;

  extern __shared__ int s_mem[];

  // [n_experts+1]
  Tensor cu_sum = make_tensor(make_smem_ptr(reinterpret_cast<int*>(s_mem)),
                              make_layout(make_shape(n_experts + 1)));
  // [n_shards+1][n_experts]
  // token_counts(0, _) = 0 is used to facilitate the prefix sum
  Tensor token_counts =
      make_tensor(make_smem_ptr(reinterpret_cast<int*>(s_mem + n_experts + 1)),
                  make_layout(make_shape(n_shards + 1, n_experts)));

  // init token counts for each expert in the shard
  for (int e_idx = 0; e_idx < n_experts; ++e_idx) {
    token_counts(curr_shard + 1, e_idx) = 0;
  }

  // calculate expert counts for each token block
  for (int i = tid; i < n_tokens; i += stride) {
    ++token_counts(curr_shard + 1, topk_ids[i]);
  }

  __syncthreads();

  // calculate the prefix sum for each expert
  // total number of tokens per expert is stored in token_counts(n_shards, _)
  if (curr_expert < n_experts) {
    token_counts(0, curr_expert) = 0;
    for (int i = 1; i <= n_shards; ++i) {
      token_counts(i, curr_expert) += token_counts(i - 1, curr_expert);
    }
  }

  __syncthreads();

  if (tid == 0) {
    // caluculate cumulative sum for each expert
    cu_sum[0] = 0;
    for (int e_idx = 1; e_idx <= n_experts; ++e_idx) {
      cu_sum[e_idx] =
          cu_sum[e_idx - 1] +
          cute::round_up(token_counts(n_shards, e_idx - 1), block_size);
    }
    *n_padded_tokens = cu_sum[n_experts];
  }

  __syncthreads();

  // each thread fills the expert id for each token block
  if (curr_expert < n_experts) {
    for (int i = cu_sum[curr_expert]; i < cu_sum[curr_expert + 1];
         i += block_size) {
      expert_ids[i / block_size] = curr_expert;
    }
  }

  for (int i = tid; i < n_tokens; i += stride) {
    const auto e_idx = topk_ids[i];
    const auto idx = token_counts(curr_shard, e_idx)++;
    sorted_token_idxes[cu_sum[e_idx] + idx] = i;
  }
}

// reduce along topk dimension, assuming contiguous memory
template <typename scalar_t, int TOPK>
__global__ void topk_sum_kernel(
    scalar_t* __restrict__ out,          // [n_tokens, dim]
    const scalar_t* __restrict__ input,  // [n_tokens, topk, dim]
    int64_t dim) {
  // one block per token
  const int64_t t_idx = blockIdx.x;
  for (int64_t i = threadIdx.x; i < dim; i += blockDim.x) {
    scalar_t sum = 0.0;
    CUTE_UNROLL
    for (int k = 0; k < TOPK; ++k) {
      sum += input[(t_idx * TOPK * dim) + (k * dim) + i];
    }
    out[(t_idx * dim) + i] = sum;
  }
}

}  // namespace

void permute_align_block(
    torch::Tensor topk_ids,  // [n_tokens, topk]
    int64_t n_experts,
    int64_t block_size,
    torch::Tensor sorted_token_idxes,  // [n_padded_permuted_tokens+]
    torch::Tensor experts_ids,         // [n_blocks+]
    torch::Tensor n_padded_tokens,     // [1]
    torch::Tensor cu_sum               // [n_experts+1]
) {
  const auto n_flatten_tokens = topk_ids.numel();
  auto* stream = at::cuda::getCurrentCUDAStream().stream();
  DISPATCH_INTEGRAL_TYPES(topk_ids.scalar_type(), "align_block_kernel", [&] {
    if (n_flatten_tokens <= 1024 && n_experts <= 64) {
      const int threads = std::max<int>(n_experts, WARP_SIZE);
      const int smem_size =
          ((threads + 1) * n_experts + (n_experts + 1)) * sizeof(int);

      align_block_kernel<scalar_t><<<1, threads, smem_size, stream>>>(
          topk_ids.data_ptr<scalar_t>(),
          sorted_token_idxes.data_ptr<scalar_t>(),
          experts_ids.data_ptr<scalar_t>(),
          n_padded_tokens.data_ptr<scalar_t>(),
          n_experts,
          block_size,
          n_flatten_tokens);
    } else {
      // each thread handles one expert
      // assert(n_experts <= 1024);
      size_t smem_size = 1024 * sizeof(int);
      cusum_kernel<scalar_t>
          <<<1, 1024, smem_size, stream>>>(topk_ids.data_ptr<scalar_t>(),
                                           experts_ids.data_ptr<scalar_t>(),
                                           n_padded_tokens.data_ptr<scalar_t>(),
                                           n_experts,
                                           block_size,
                                           n_flatten_tokens,
                                           cu_sum.data_ptr<scalar_t>());

      constexpr int threads = 256;
      int n_blocks = cute::ceil_div(n_flatten_tokens, threads);
      n_blocks = std::min(n_blocks, 65535);
      row_id_map_kernel<scalar_t><<<n_blocks, threads, 0, stream>>>(
          topk_ids.data_ptr<scalar_t>(),
          sorted_token_idxes.data_ptr<scalar_t>(),
          cu_sum.data_ptr<scalar_t>(),
          n_flatten_tokens);
    }
  });
}

void sum_out(const torch::Tensor& input,  // [n_tokens, topk, dim]
             torch::Tensor& output)       // [n_tokens, dim]
{
  const auto n_tokens = input.size(0);
  const auto topk = input.size(1);
  const auto dim = input.size(2);

  // one block per token
  const auto threads = std::min<int>(dim, 1024);
  auto* stream = at::cuda::getCurrentCUDAStream().stream();

#define DISPATCH_TOPK_SUM_KERNEL_CASE(TOPK)                                    \
  case TOPK: {                                                                 \
    DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_kernel", [&] {           \
      topk_sum_kernel<scalar_t, TOPK><<<n_tokens, threads, 0, stream>>>(       \
          output.data_ptr<scalar_t>(), input.const_data_ptr<scalar_t>(), dim); \
    });                                                                        \
    break;                                                                     \
  }

  switch (topk) {
    DISPATCH_TOPK_SUM_KERNEL_CASE(2);
    DISPATCH_TOPK_SUM_KERNEL_CASE(3);
    DISPATCH_TOPK_SUM_KERNEL_CASE(4);
    DISPATCH_TOPK_SUM_KERNEL_CASE(8);
    default:
      // use torch::sum_out for other cases
      torch::sum_out(output, input, /*dim=*/1);
      break;
  }
}

}  // namespace llm::kernel::moe
