#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include "dispatch.h"
#include "kv_cache_kernels.h"
namespace llm::kernel {

template <typename T>
__global__ void set_kv_cache_kernel(
    const int* __restrict__ slot_ids,  // [n_tokens]
    const T* __restrict__ keys,        // [n_tokens, n_heads, head_dim]
    const T* __restrict__ values,      // [n_tokens, n_heads, head_dim]
    T* __restrict__ key_cache,
    T* __restrict__ value_cache,
    int kv_stride,
    int n_kv_heads,
    int head_dim,
    int block_size) {
  // block/token index
  const int64_t bid = blockIdx.x;
  // which slot to write to
  const int64_t slot_id = slot_ids[bid];
  // block index
  const int64_t block_idx = slot_id / block_size;
  // offset within block
  const int64_t block_offset = slot_id % block_size;

  // base index for the block in cache
  const int64_t block_base_idx = block_idx * block_size * n_kv_heads * head_dim;

  // copy value one by one for the token
  for (int i = threadIdx.x; i < n_kv_heads * head_dim; i += blockDim.x) {
    const int64_t src_idx = bid * kv_stride + i;

    // cache: [n_blocks, block_size, n_heads, head_dim]
    const int64_t head_base_idx =
        block_base_idx + block_offset * n_kv_heads * head_dim;

    // which head to write to
    const int head_idx = i / head_dim;
    // which dim within head to write to
    const int head_offset = i % head_dim;
    const int64_t dst_idx = head_base_idx + head_idx * head_dim + head_offset;

    key_cache[dst_idx] = keys[src_idx];
    value_cache[dst_idx] = values[src_idx];
  }
}

void set_kv_cache(
    const torch::Tensor& slot_ids,  // [n_tokens]
    const torch::Tensor& keys,      // [n_tokens, n_kv_heads, head_dim]
    const torch::Tensor& values,    // [n_tokens, n_kv_heads, head_dim]
    torch::Tensor& key_cache,       // [n_blocks, block_size, n_heads, head_dim]
    torch::Tensor& value_cache) {
  const int n_tokens = keys.size(0);
  const int n_kv_heads = keys.size(-2);
  const int head_dim = keys.size(-1);
  const int block_size = key_cache.size(-3);
  const int kv_stride = keys.stride(0);
  const int n = n_kv_heads * head_dim;

  dim3 grid(n_tokens);
  dim3 block(std::min(n, 1024));
  DISPATCH_FLOATING_TYPES(keys.scalar_type(), "set_kv_cache_kernel", [&] {
    set_kv_cache_kernel<scalar_t>
        <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            slot_ids.data_ptr<int>(),
            keys.data_ptr<scalar_t>(),
            values.data_ptr<scalar_t>(),
            key_cache.data_ptr<scalar_t>(),
            value_cache.data_ptr<scalar_t>(),
            kv_stride,
            n_kv_heads,
            head_dim,
            block_size);
  });
}

}  // namespace llm::kernel
