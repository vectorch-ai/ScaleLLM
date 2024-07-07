#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include "dispatch.h"
#include "pos_embedding_kernels.h"

namespace llm::kernel {

template <typename T>
struct RotaryEmbedding {
  // apply rotary embedding to data on position idx
  // x -> x * cos - y * sin
  // y -> x * sin + y * cos
  static __device__ __forceinline__ void apply(T* __restrict__ data,
                                               const T* __restrict__ cos,
                                               const T* __restrict__ sin,
                                               int idx,
                                               int n,
                                               bool interleaved) {
    // interleaved: x = data[2 * idx], y = data[2 * idx + 1]
    // rotated: x = data[idx], y = data[idx + rotary_dim / 2]
    const int x_idx = interleaved ? 2 * idx : idx;
    const int y_idx = interleaved ? 2 * idx + 1 : idx + n;
    const T x = data[x_idx];
    const T y = data[y_idx];
    const T c = cos[idx];
    const T s = sin[idx];
    data[x_idx] = x * c - y * s;
    data[y_idx] = x * s + y * c;
  }
};

// inplace update query and key
template <typename T>
__global__ void rotary_embedding_kernel(
    T* __restrict__ querys,             // [n_tokens, n_heads, head_dim]
    T* __restrict__ keys,               // [n_tokens, n_kv_heads, head_dim]
    const int* __restrict__ positions,  // [n_tokens]
    const T* __restrict__ cos_sin,      // [max_positions, 2, rotary_dim/2]
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t n_heads,
    int64_t n_kv_heads,
    int64_t q_stride,
    int64_t k_stride,
    bool interleaved) {
  const int tidx = threadIdx.x;
  const int bidx = blockIdx.x;

  // figure out cos sin base ptr for the token
  const int64_t n = rotary_dim / 2;
  const T* cos_sin_base = cos_sin + positions[bidx] * rotary_dim;
  const T* cos = cos_sin_base;
  const T* sin = cos_sin_base + n;

  // apply rotary embedding to query head by head
  // q base ptr for the token
  T* q_base = querys + bidx * q_stride;
  for (int64_t i = tidx; i < n_heads * n; i += blockDim.x) {
    // head idx
    const int64_t h_idx = i / n;
    // rotary idx within head
    const int64_t r_idx = i % n;
    // q ptr for the head
    T* q = q_base + h_idx * head_dim;
    RotaryEmbedding<T>::apply(q, cos, sin, r_idx, n, interleaved);
  }

  // apply rotary embedding to key head by head
  // k base ptr for the token
  T* k_base = keys + bidx * k_stride;
  for (int64_t i = tidx; i < n_kv_heads * n; i += blockDim.x) {
    // head idx
    const int64_t h_idx = i / n;
    // rotary idx within head
    const int64_t r_idx = i % n;
    // k ptr for the head
    T* k = k_base + h_idx * head_dim;
    RotaryEmbedding<T>::apply(k, cos, sin, r_idx, n, interleaved);
  }
}

// apply rotary embedding to query and key inplace
void apply_rotary_pos_emb(
    torch::Tensor& querys,           // [n_tokens, n_heads, head_dim]
    torch::Tensor& keys,             // [n_tokens, n_kv_heads, head_dim]
    const torch::Tensor& positions,  // [n_tokens]
    const torch::Tensor& cos_sin,    // [max_positions, 2, rotary_dim/2]
    int rotary_dim,
    bool interleaved) {
  // keys and values should be continuous at n_kv_heads and head_dim dims
  CHECK(querys.stride(-1) == 1 && querys.stride(-2) == querys.size(-1));
  CHECK(keys.stride(-1) == 1 && keys.stride(-2) == keys.size(-1));

  const int64_t n_tokens = querys.size(-3);
  const int64_t n_heads = querys.size(-2);
  const int64_t n_kv_heads = keys.size(-2);
  const int64_t head_dim = querys.size(-1);
  const int64_t q_stride = querys.stride(-3);
  const int64_t k_stride = keys.stride(-3);

  const dim3 grid(n_tokens);
  const dim3 block(std::min<int>(1024, n_heads * rotary_dim) / 2);
  DISPATCH_FLOATING_TYPES(querys.scalar_type(), "rotary_embedding_kernel", [&] {
    rotary_embedding_kernel<scalar_t>
        <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            querys.data_ptr<scalar_t>(),
            keys.data_ptr<scalar_t>(),
            positions.data_ptr<int>(),
            cos_sin.data_ptr<scalar_t>(),
            head_dim,
            rotary_dim,
            n_heads,
            n_kv_heads,
            q_stride,
            k_stride,
            interleaved);
  });
}

}  // namespace llm::kernel
