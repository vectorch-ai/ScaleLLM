#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include "dispatch.h"
#include "kv_cache_kernels.h"
#include "sentencepiece/darts.h"
namespace llm::kernel {

template <typename T>
__global__ void set_kv_cache_kernel(
    const int* __restrict__ slot_ids,  // [n_tokens]
    const T* __restrict__ keys,        // [n_tokens, n_heads, head_dim]
    const T* __restrict__ values,      // [n_tokens, n_heads, head_dim]
    T* __restrict__ key_cache,
    T* __restrict__ value_cache,
    int key_stride,
    int value_stride,
    int n_heads,
    int head_dim,
    int block_size,
    int x) {
  // block/token index
  const int bid = blockIdx.x;
  // which slot to write to
  const int slot_id = slot_ids[bid];
  // block index
  const int block_idx = slot_id / block_size;
  // offset within block
  const int block_offset = slot_id % block_size;

  // base index for the block in cache
  const int block_base_idx = block_idx * n_heads * head_dim * block_size;

  // copy value one by one for the token
  for (int i = threadIdx.x; i < n_heads * head_dim; i += blockDim.x) {
    const int src_key_idx = bid * key_stride + i;
    const int src_value_idx = bid * value_stride + i;

    // which head to write to
    const int head_idx = i / head_dim;
    // which dim within head to write to
    const int head_offset = i % head_dim;

    const int x_idx = head_offset / x;
    const int x_offset = head_offset % x;

    // key cache: [n_blocks, n_heads, head_dim/x, block_size, x]
    const int head_base_idx = block_base_idx + head_idx * head_dim * block_size;
    const int dst_key_idx =
        head_base_idx + x_idx * block_size * x + block_offset * x + x_offset;

    // value cache: [n_blocks, n_heads, head_dim, block_size]
    const int dst_value_idx =
        head_base_idx + head_offset * block_size + block_offset;

    key_cache[dst_key_idx] = __ldg(&keys[src_key_idx]);
    value_cache[dst_value_idx] = __ldg(&values[src_value_idx]);
  }
}

void set_kv_cache(
    const torch::Tensor& slot_ids,  // [n_tokens]
    const torch::Tensor& keys,      // [n_tokens, n_heads, head_dim]
    const torch::Tensor& values,    // [n_tokens, n_heads, head_dim]
    torch::Tensor& key_cache,  // [n_blocks, n_heads, head_dim/x, block_size, x]
    torch::Tensor& value_cache) {
  const int n_tokens = keys.size(0);
  const int n_heads = keys.size(1);
  const int head_dim = keys.size(2);
  const int block_size = key_cache.size(3);
  const int x = key_cache.size(4);
  const int key_stride = keys.stride(0);
  const int value_stride = values.stride(0);
  const int n = n_heads * head_dim;

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
            key_stride,
            value_stride,
            n_heads,
            head_dim,
            block_size,
            x);
  });
}

}  // namespace llm::kernel
