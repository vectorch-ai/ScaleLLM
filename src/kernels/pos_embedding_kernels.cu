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
  static __device__ __forceinline__ T apply(T* __restrict__ data,
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
    T* __restrict__ query,              // [n_tokens, n_heads, head_dim]
    T* __restrict__ key,                // [n_tokens, n_kv_heads, head_dim]
    const int* __restrict__ positions,  // [n_tokens]
    const T* __restrict__ cos_sin,      // [max_positions, 2, rotary_dim/2]
    int head_dim,
    int rotary_dim,
    int n_heads,
    int n_kv_heads,
    int q_stride,
    int k_stride,
    bool interleaved) {
  const int tidx = threadIdx.x;
  const int bidx = blockIdx.x;

  // figure out cos sin base ptr for the token
  const int n = rotary_dim / 2;
  const T* cos_sin_base = cos_sin + positions[bidx] * rotary_dim;
  const T* cos = cos_sin_base;
  const T* sin = cos_sin_base + n;

  // apply rotary embedding to query head by head
  // q base ptr for the token
  T* q_base = query + bidx * q_stride;
  for (int i = tidx; i < n_heads * n; i += blockDim.x) {
    // head idx
    const int h_idx = i / n;
    // rotary idx within head
    const int r_idx = i % n;
    // q ptr for the head
    T* q = q_base + h_idx * head_dim;
    RotaryEmbedding<T>::apply(q, cos, sin, r_idx, n, interleaved);
  }

  // apply rotary embedding to key head by head
  // k base ptr for the token
  T* k_base = key + bidx * k_stride;
  for (int i = tidx; i < n_kv_heads * n; i += blockDim.x) {
    // head idx
    const int h_idx = i / n;
    // rotary idx within head
    const int r_idx = i % n;
    // k ptr for the head
    T* k = k_base + h_idx * head_dim;
    RotaryEmbedding<T>::apply(k, cos, sin, r_idx, n, interleaved);
  }
}

// apply rotary embedding to query and key inplace
void apply_rotary_pos_emb(
    torch::Tensor& query,            // [..., n_heads, head_dim]
    torch::Tensor& key,              // [..., n_kv_heads, head_dim]
    const torch::Tensor& positions,  // [...]
    const torch::Tensor& cos_sin,    // [max_positions, 2, rotary_dim/2]
    int rotary_dim,
    bool interleaved) {
  DCHECK(query.is_cuda()) << "query must be on gpu";
  DCHECK(key.is_cuda()) << "key must be on gpu";
  DCHECK(query.dim() == 3) << "query must be 3d";
  DCHECK(key.dim() == 3) << "key must be 3d";

  const int n_tokens = query.numel() / query.size(-1);
  const int n_heads = query.size(-2);
  const int n_kv_heads = key.size(-2);
  const int head_dim = query.size(-1);
  const int q_stride = query.stride(-2);
  const int k_stride = key.stride(-2);

  const dim3 grid(n_tokens);
  const dim3 block(std::min(1024, n_heads * rotary_dim) / 2);
  DISPATCH_FLOATING_TYPES(query.scalar_type(), "rotary_embedding_kernel", [&] {
    rotary_embedding_kernel<scalar_t>
        <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            query.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(),
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
