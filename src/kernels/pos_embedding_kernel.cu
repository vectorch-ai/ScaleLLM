#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "reduce_kernel_utils.cuh"

namespace llm::kernel {

// inplace update query and key
template <typename scalar_t>
__global__ void interleaved_rotary_embedding_kernel(
    const scalar_t* __restrict__ query,  // [num_tokens, num_heads, head_dim]
    const scalar_t* __restrict__ key,    // [num_tokens, num_kv_heads, head_dim]
    const int64_t* __restrict__ positions,       // [num_tokens]
    const scalar_t* __restrict__ cos_sin_cache,  // [max_seq_len, 2 * rotary_dim]
    const uint32_t head_dim,
    const uint32_t rotary_dim) {
  // each thread block handles one token
  const auto token_idx = blockIdx.x;

  // figure out cache index for the token
  const auto pos = positions[token_idx];
  const scalar_t* cache = cos_sin_cache + pos * 2 * head_dim;

  // each thread process one position in the dim
}

}  // namespace llm::kernel
