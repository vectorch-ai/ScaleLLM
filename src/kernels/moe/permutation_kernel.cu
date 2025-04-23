// Adapted from
// https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/permutation/permutation.cu
#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include <cub/cub.cuh>
#include <cute/numeric/numeric_types.hpp>

// clang-format off
// for exmple: n_tokens = 2, n_experts = 8, topk = 2
// f_idx: idx in flatten indices
// p_idx: idx in permuted tokens
// k_idx: topk idx
// t_idx: token idx
// row_id_map: [topk, n_tokens] => idx in permuted tokens
//  ____________________________________________________________________________________________________________________________
// |                 |     flatten indices         |        sort flatten indices          |           row_id_map                |
// |    Steps        |   sort by (tokens, topk)    |        by (experts, tokens)          |     sort by (topk, tokens)          |
// |_________________|_____________________________|______________________________________|_____________________________________|
// |                 |    [n_tokens * topk]        |     [n_tokens * topk] => f_idx       |      [topk, n_tokens] => p_idx      |
// |     Dim         |                             |   f_idx: idx in flatten indices      |    p_idx: idx in permuted tokens    |
// |_________________|_____________________________|______________________________________|_____________________________________|
// |                 |                             |                                      |                                     |
// |      top0, top1 |   f_idx: | 0 | 1 | 2 | 3 |  |   p_idx: |  0  |  1  |  2  |  3  |   |     idx: |  0  |  1  |  2  |  3  |  |
// | t0 -> [e2, e1]  | experts: | 2 | 1 | 2 | 5 |  |   f_idx: |  1  |  0  |  2  |  3  |   |   p_idx: |  1  |  2  |  0  |  3  |  |
// | t1 -> [e2, e5]  |  tokens: |   t0  |   t1  |  |  tokens: |  t0 |  t0 |  t1 |  t1 |   |   f_idx: |  0  |  2  |  1  |  3  |  |
// |                 |                             | experts: |  e1 |     e2    |  e5 |   | experts: |  e2 |  e2 |  e1 |  e5 |  |
// |                 |                             |                                      |  tokens: |  t0 |  t1 |  t0 |  t1 |  |
// |                 |                             |                                      |    topk: |    top0   |    top1   |  |
// |_________________|_____________________________|______________________________________|_____________________________________|
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

// wrapper for cub::DeviceRadixSort::SortPairs with type
void radix_sort_pairs(void* temp_storage,
                      size_t* temp_storage_bytes,
                      const int* keys_in,
                      int* keys_out,
                      const int* values_in,
                      int* values_out,
                      size_t num_items) {
  cub::DeviceRadixSort::SortPairs(temp_storage,
                                  *temp_storage_bytes,
                                  keys_in,
                                  keys_out,
                                  values_in,
                                  values_out,
                                  num_items);
}

// build a row_id_map that maps [topk, n_tokens] to the index in permuted tokens
__global__ void permute_row_id_map(
    const int* sorted_row_id,  // [n_permuted_tokens]
    int* row_id_map,           // [topk, n_tokens]
    const int n_tokens,
    const int topk) {
  // row_id_map[topK][n_tokens]
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;

  // idx in permuted tokens
  const int p_idx = bid * blockDim.x + tid;
  const int n_permuted_tokens = n_tokens * topk;

  if (p_idx >= n_permuted_tokens) {
    return;
  }

  // idx in flattened indices
  const int f_idx = sorted_row_id[p_idx];
  // token idx: each token has topk experts in flattened indices
  const int t_idx = f_idx / topk;
  // topk idx: idx in topk experts for the token
  const int k_idx = f_idx % topk;

  // row_id_map: [topk, n_tokens] => idx in permuted tokens
  row_id_map[(k_idx * n_tokens) + t_idx] = p_idx;
}

template <typename T,
          int kTopK>
__global__ void permute_kernel(
    const T* tokens,        // [n_tokens, dim]
    T* permuted_tokens,     // [n_permuted_tokens, dim]
    const int* row_id_map,  // [topk, n_tokens] => p_idx
    const int n_tokens,
    const int topk,
    const int dim) {
  // one block corresponds to one token
  const int token_idx = blockIdx.x;
  const int tid = threadIdx.x;

  // frag for load/store
  float4 frag_ls;

  static constexpr int kFragSize = 16 / sizeof(T);
  // tokens: [n_tokens, dim]
  const T* token_base = tokens + token_idx * dim;
  for (int i = tid * kFragSize; i < dim; i += blockDim.x * kFragSize) {
    // load fragment into frag_ls (float4)
    frag_ls = __ldlu(reinterpret_cast<const float4*>(token_base + i));

    int idx = token_idx;
    for (int k = 0; k < kTopK; k++) {
      if (k == topk) {
        break;
      }

      // row_id_map: [topk, n_tokens] => idx in permuted tokens
      const int p_idx = row_id_map[idx];
      // move to next k
      idx += n_tokens;

      // store back to permuted_tokens: [n_permuted_tokens, dim]
      T* permuted_token_base = permuted_tokens + p_idx * dim;
      *reinterpret_cast<float4*>(permuted_token_base + i) = frag_ls;
    }
  }
}

template <typename T>
__global__ void unpermute_kernel(
    const T* permuted_tokens,  // [n_permuted_tokens, dim]
    T* tokens,                 // [n_tokens, dim]
    const int* row_id_map,     // [topk, n_tokens] => idx in permuted tokens
    const T* probs,            // [n_tokens, topk]
    const int n_tokens,
    const int topk,
    const int dim) {
  extern __shared__ int8_t s_mem[];
  // [topk] probs for the token
  T* s_probs = reinterpret_cast<T*>(s_mem);

  // each block corresponds to one token
  const int t_idx = blockIdx.x;
  const int tid = threadIdx.x;

  // load prob into shared memory for the token
  // let first topk thread to load probs
  for (int i = tid; i < topk; i += blockDim.x * blockDim.y) {
    s_probs[i] = probs[(t_idx * topk) + i];
  }
  __syncthreads();

  // float4 for load and store
  float4 frag_ls;
  T* frag_ls_ptr = reinterpret_cast<T*>(&frag_ls);

  static constexpr int kFragSize = 16 / sizeof(T);
  for (int i = tid * kFragSize; i < dim; i += blockDim.x * kFragSize) {
    T frag_sum[kFragSize];
    for (int d = 0; d < kFragSize; d++) {
      frag_sum[d] = T(0.0f);
    }

    // sum over topk
    for (int k_idx = 0; k_idx < topk; k_idx++) {
      const int p_idx = row_id_map[(k_idx * n_tokens) + t_idx];
      const T* permuted_token_base = permuted_tokens + p_idx * dim;
      // load fragment into frag_ls (float4)
      frag_ls =
          __ldlu(reinterpret_cast<const float4*>(permuted_token_base + i));

      T frag[kFragSize];
      for (int d = 0; d < kFragSize; d++) {
        frag[d] = frag_ls_ptr[d];
      }

      // apply probs & sum
      for (int d = 0; d < kFragSize; d++) {
        frag[d] *= s_probs[k_idx];
        frag_sum[d] += frag[d];
      }
    }

    // write back to frag_ls (float4)
    for (int d = 0; d < kFragSize; d++) {
      frag_ls_ptr[d] = frag_sum[d];
    }

    // store back to tokens: [n_tokens, dim]
    T* token_base = tokens + t_idx * dim;
    *reinterpret_cast<float4*>(token_base + i) = frag_ls;
  }
}

template <typename T>
void launch_permute_kernel(
    const T* tokens,           // [n_tokens, dim]
    T* permuted_tokens,        // [n_permuted_tokens, dim]
    const int* sorted_row_id,  // [n_permuted_tokens] -> flattened index
    int* row_id_map,           // [topk, n_tokens] -> idx in permuted tokens
    const int n_tokens,
    const int topk,
    const int dim,
    cudaStream_t stream) {
  const int n_permuted_tokens = n_tokens * topk;
  int threads = 64;
  int blocks = (n_permuted_tokens + threads - 1) / threads;
  permute_row_id_map<<<blocks, threads, 0, stream>>>(
      sorted_row_id, row_id_map, n_tokens, topk);

  // use 128-bit load/store
  constexpr int kFragSize = 16 / sizeof(T);
  // assert(dim % kFragSize == 0);

  // one block per source token
  blocks = n_tokens;
  threads = std::min(dim / kFragSize, 1024);
  // assert(topk <= 128);
  permute_kernel<T, /*TOPK=*/128><<<blocks, threads, 0, stream>>>(
      tokens, permuted_tokens, row_id_map, n_tokens, topk, dim);
}

template <typename T>
void launch_unpermute_kernel(
    const T* permuted_tokens,  // [n_permuted_tokens, dim]
    T* tokens,                 // [n_tokens, dim]
    int* row_id_map,           // [topk, n_tokens] => dst row
    const T* prob,             // [n_tokens, topk]
    const int n_tokens,
    const int topk,
    const int dim,
    cudaStream_t stream) {
  // use 128-bit load/store
  constexpr int kFragSize = 16 / sizeof(T);
  // assert(dim % kFragSize == 0);

  // each block corresponds to one token
  int blocks = n_tokens;
  // up to 1024 threads per block
  int threads = std::min(dim / kFragSize, 1024);
  size_t smem_bytes = topk * sizeof(T);

  // unpermute_topK fwd
  unpermute_kernel<T><<<blocks, threads, smem_bytes, stream>>>(
      permuted_tokens, tokens, row_id_map, prob, n_tokens, topk, dim);
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> permute(
    torch::Tensor tokens,  // [n_tokens, dim]
    torch::Tensor indices  // [n_tokens, topk]
) {
  const auto n_tokens = tokens.size(0);
  const auto dim = tokens.size(1);
  const auto topk = indices.size(1);

  const auto n_permuted_tokens = n_tokens * topk;
  const auto options = tokens.options();

  // TODO: cache workspace to avoid reallocating
  // calculate the size of temporary storage
  size_t temp_storage_bytes = 0;
  radix_sort_pairs(nullptr,
                   &temp_storage_bytes,
                   nullptr,
                   nullptr,
                   nullptr,
                   nullptr,
                   n_permuted_tokens);
  // allocate temporary storage
  auto temp_storage =
      torch::empty(temp_storage_bytes, options.dtype(torch::kInt8));

  const auto int32_options = options.dtype(torch::kInt32);
  auto sorted_indices = torch::zeros(n_permuted_tokens, int32_options);
  auto row_id = torch::range(0, n_permuted_tokens - 1, 1, int32_options);
  auto sorted_row_id = torch::zeros(n_permuted_tokens, int32_options);

  const int* indices_ptr = indices.const_data_ptr<int>();
  const int* row_id_ptr = row_id.const_data_ptr<int>();
  int* sorted_indices_ptr = sorted_indices.data_ptr<int>();
  int* sorted_row_id_ptr = sorted_row_id.data_ptr<int>();
  void* d_temp_storage = temp_storage.data_ptr();

  radix_sort_pairs(d_temp_storage,
                   &temp_storage_bytes,
                   indices_ptr,
                   sorted_indices_ptr,
                   row_id_ptr,
                   sorted_row_id_ptr,
                   n_permuted_tokens);

  const auto type = tokens.scalar_type();

  auto permuted_tokens = torch::empty({n_permuted_tokens, dim},
                                      torch::dtype(type).device(torch::kCUDA));
  auto row_id_map = torch::empty(
      {n_tokens * topk}, torch::dtype(torch::kInt32).device(torch::kCUDA));

  auto* stream = at::cuda::getCurrentCUDAStream().stream();

#define LAUNCH_PERMUTE_KERNEL(DType)                             \
  launch_permute_kernel<DType>(const_data_ptr<DType>(tokens),    \
                               data_ptr<DType>(permuted_tokens), \
                               sorted_row_id_ptr,                \
                               row_id_map.data_ptr<int>(),       \
                               n_tokens,                         \
                               topk,                             \
                               dim,                              \
                               stream);

  switch (type) {
    case torch::ScalarType::Float: {
      LAUNCH_PERMUTE_KERNEL(float);
      break;
    }
    case torch::ScalarType::Half: {
      LAUNCH_PERMUTE_KERNEL(cute::half_t);
      break;
    }
    case torch::ScalarType::BFloat16: {
      LAUNCH_PERMUTE_KERNEL(cute::bfloat16_t);
      break;
    }
    default:
      CHECK(false) << "Unsupported tensor type: " << type;
  }

  return {permuted_tokens, row_id_map};
}

torch::Tensor unpermute(
    torch::Tensor permuted_tokens,  // [n_permuted_tokens, dim]
    torch::Tensor row_id_map,       // [topk, n_tokens] => dst row
    torch::Tensor probs,            // [n_tokens, topk]
    int64_t n_tokens,
    int64_t topk) {
  const auto dim = permuted_tokens.size(1);
  const auto type = permuted_tokens.scalar_type();

  // [n_tokens, dim]
  auto tokens = torch::empty(
      {n_tokens, dim},
      torch::dtype(type).device(torch::kCUDA).requires_grad(false));

  auto* stream = at::cuda::getCurrentCUDAStream().stream();

#define LAUNCH_UNPERMUTE_KERNEL(DType)                                   \
  launch_unpermute_kernel<DType>(const_data_ptr<DType>(permuted_tokens), \
                                 data_ptr<DType>(tokens),                \
                                 row_id_map.data_ptr<int>(),             \
                                 const_data_ptr<DType>(probs),           \
                                 n_tokens,                               \
                                 topk,                                   \
                                 dim,                                    \
                                 stream);

  switch (type) {
    case torch::ScalarType::Float: {
      LAUNCH_UNPERMUTE_KERNEL(float);
      break;
    }
    case torch::ScalarType::Half: {
      LAUNCH_UNPERMUTE_KERNEL(cute::half_t);
      break;
    }
    case torch::ScalarType::BFloat16: {
      LAUNCH_UNPERMUTE_KERNEL(cute::bfloat16_t);
      break;
    }
    default:
      CHECK(false) << "Unsupported tensor type: " << type;
  }

  return tokens;
}

}  // namespace llm::kernel::moe
