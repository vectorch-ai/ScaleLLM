// Adapted from
// https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/permutation/permutation.cu
#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include <cub/cub.cuh>
#include <cute/config.hpp>
#include <cute/numeric/numeric_types.hpp>

// clang-format off
// for exmple: n_tokens = 4, n_experts = 4, topk = 2, block_size=2
// f_idx: idx in flatten indices
// p_idx: idx in permuted tokens
// k_idx: topk idx
// t_idx: token idx
// row_id_map: [topk, n_tokens] => idx in permuted tokens
//  _______________________________________________________________________________________________________
// |                         |                         |                         |                         |
// |    Steps                |        routing_map      |   row_id_map -> cu_sum  |   row_id_map -> p_idx   |
// |                         |   [n_tokens, n_experts] |  [n_experts, n_tokens]  |  [n_experts, n_tokens]  |
// |_________________________|_________________________|_________________________|_________________________|
// |                         |                         |                         |                         |
// |                         |           e_idx         |           t_idx         |           t_idx         |
// |      top0, top1         |          0 1 2 3        |          0 1 | 2 3      |          0 1 | 2 3      |
// | t0 -> [e2, e1]          |    t0  | 0 1 1 0 |      |    e0  | x 1 | 1 x |    |    e0  | x 0 | 1 x |    |
// | t1 -> [e1, e0]          |    t1  | 1 1 0 0 |      |    e1  | 1 2 | 1 x |    |    e1  | 2 3 | 4 x |    |
// | t2 -> [e0, e1]          |    t2  | 1 1 0 0 |      |    e2  | 1 x | x 1 |    |    e2  | 5 x | x 6 |    |
// | t3 -> [e2, e3]          |    t3  | 0 0 1 1 |      |    e3  | x x | x 1 |    |    e3  | x x | x 7 |    |
// |_________________________|_________________________|_________________________|_________________________|
// |                         |                         |        block_sum        |      block_cu_sum       |
// |                         |                         |    b0     1  |  1       |  c_b0     0  |  1       |
// |                         |                         |    b1     2  |  1       |  c_b1     2  |  4       |
// |                         |                         |    b2     1  |  1       |  c_b2     5  |  6       |
// |                         |                         |    b3     0  |  1       |  c_b3     7  |  7       |
// |_________________________|_________________________|_________________________|_________________________|
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

// (grid, block) = (n_experts, min(n_blocks, 1024))
template <int BLOCK_SIZE>
__global__ void block_sum_kernel(
    const bool* routing_map,  // [n_tokens, n_experts]
    int* row_id_map,          // [n_experts, n_tokens]
    int* block_sum,           // [n_experts, n_blocks]
    const int n_tokens,
    const int n_experts,
    const int n_blocks) {
  // expert idx
  const int e_idx = blockIdx.x;
  // start token idx
  const int tid = threadIdx.x;

  // process each token block
  for (int b = tid; b < n_blocks; b += blockDim.x) {
    // block start token idx
    const int t_base = b * BLOCK_SIZE;
    int sum = 0;
    // process each token in the block
    for (int i = 0; i < BLOCK_SIZE; ++i) {
      const int t_idx = t_base + i;
      if (t_idx < n_tokens) {
        // routing_map: [n_tokens, n_experts]
        const auto val = routing_map[(t_idx * n_experts) + e_idx];
        // row_id_map: [n_experts, n_tokens]
        row_id_map[(e_idx * n_tokens) + t_idx] = val ? ++sum : 0;
      }
    }
    // block_sum: [n_experts, n_blocks]
    block_sum[(e_idx * n_blocks) + b] = sum;
  }
}

// (grid, block) = (n_experts, min(n_blocks, 1024))
template <int BLOCK_SIZE>
__global__ void row_id_map_kernel(
    const int* block_sum,  // [n_experts, n_blocks]
    int* row_id_map,       // [n_experts, n_tokens]
    const int n_tokens,
    const int n_blocks) {
  // expert idx
  const int e_idx = blockIdx.x;
  // start token idx
  const int tid = threadIdx.x;
  // process each token block
  for (int b = tid; b < n_blocks; b += blockDim.x) {
    const int g_b = n_blocks * e_idx + b;
    int cu_sum = 0;
    for (int i = 0; i < g_b; ++i) {
      cu_sum += block_sum[i];
    }

    // block start token idx
    const int t_base = b * BLOCK_SIZE;
    // process each token in the block
    for (int i = 0; i < BLOCK_SIZE; ++i) {
      const int t_idx = t_base + i;
      if (t_idx < n_tokens) {
        // row_id_map: [n_experts, n_tokens]
        const int idx = (e_idx * n_tokens) + t_idx;
        const auto val = row_id_map[idx];
        row_id_map[idx] = val ? cu_sum + val - 1 : -1;
      }
    }
  }
}

template <typename T>
__global__ void permute_kernel(
    const T* tokens,        // [n_tokens, dim]
    T* permuted_tokens,     // [n_permuted_tokens, dim]
    const int* row_id_map,  // [n_experts, n_tokens] => p_idx
    const int n_tokens,
    const int n_experts,
    const int dim) {
  // one block corresponds to one token
  const int t_idx = blockIdx.x;
  const int tid = threadIdx.x;

  // frag for load/store
  float4 frag_ls;

  static constexpr int kFragSize = 16 / sizeof(T);
  // tokens: [n_tokens, dim]
  const T* token_base = tokens + t_idx * dim;
  for (int i = tid * kFragSize; i < dim; i += blockDim.x * kFragSize) {
    // load fragment into frag_ls (float4)
    frag_ls = __ldlu(reinterpret_cast<const float4*>(token_base + i));

    // broadcast to all experts
    for (int e_idx = 0; e_idx < n_experts; ++e_idx) {
      // row_id_map: [n_experts, n_tokens] => idx in permuted tokens
      const auto p_idx = row_id_map[(e_idx * n_tokens) + t_idx];
      if (p_idx != -1) {
        // store back to permuted_tokens: [n_permuted_tokens, dim]
        T* permuted_token_base = permuted_tokens + p_idx * dim;
        *reinterpret_cast<float4*>(permuted_token_base + i) = frag_ls;
      }
    }
  }
}

template <typename T>
__global__ void unpermute_kernel(
    const T* permuted_tokens,  // [n_permuted_tokens, dim]
    T* tokens,                 // [n_tokens, dim]
    const int* row_id_map,  // [n_experts, n_tokens] => idx in permuted tokens
    const T* probs,         // [n_tokens, n_experts]
    const int n_tokens,
    const int n_experts,
    const int dim) {
  extern __shared__ int8_t s_mem[];
  // [n_experts] probs for the token
  T* s_probs = reinterpret_cast<T*>(s_mem);

  // each block corresponds to one token
  const int t_idx = blockIdx.x;
  const int tid = threadIdx.x;

  // load prob into shared memory for the token
  for (int i = tid; i < n_experts; i += blockDim.x) {
    s_probs[i] = probs[(t_idx * n_experts) + i];
  }
  __syncthreads();

  // float4 for load and store
  float4 frag_ls;
  T* frag_ls_ptr = reinterpret_cast<T*>(&frag_ls);

  static constexpr int kFragSize = 16 / sizeof(T);
  for (int i = tid * kFragSize; i < dim; i += blockDim.x * kFragSize) {
    T frag_sum[kFragSize] = {T(0.0f)};

    // sum over experts
    for (int e_idx = 0; e_idx < n_experts; ++e_idx) {
      // row_id_map: [n_experts, n_tokens] => idx in permuted tokens
      const int p_idx = row_id_map[(e_idx * n_tokens) + t_idx];
      if (p_idx != -1) {
        const T* permuted_token_base = permuted_tokens + p_idx * dim;
        // load fragment into frag_ls (float4)
        frag_ls =
            __ldlu(reinterpret_cast<const float4*>(permuted_token_base + i));

        // apply probs & sum
        const auto prob = s_probs[e_idx];
        CUTE_UNROLL
        for (int d = 0; d < kFragSize; ++d) {
          frag_sum[d] += (frag_ls_ptr[d] * prob);
        }
      }
    }

    // store back to tokens: [n_tokens, dim]
    T* token_base = tokens + t_idx * dim;
    *reinterpret_cast<float4*>(token_base + i) =
        *reinterpret_cast<float4*>(frag_sum);
  }
}

template <typename T, int BLOCK_SIZE>
void launch_permute_kernel(
    const T* tokens,          // [n_tokens, dim]
    T* permuted_tokens,       // [n_permuted_tokens, dim]
    const bool* routing_map,  // [n_tokens, n_experts] bool
    int* block_sum,           // [n_experts, n_blocks] int
    int* row_id_map,          // [n_experts, n_tokens] int
    const int n_tokens,
    const int n_experts,
    const int topk,
    const int dim,
    const int n_blocks,
    cudaStream_t stream) {
  // step1: transpose routing_map to [n_experts, n_tokens] and calculate block
  // sum for each expert
  block_sum_kernel<BLOCK_SIZE><<<n_experts, n_blocks, 0, stream>>>(
      routing_map, row_id_map, block_sum, n_tokens, n_experts, n_blocks);

  // step2: calculate index in permuted tokens for each token
  // launch_row_id_kernel;
  row_id_map_kernel<BLOCK_SIZE><<<n_experts, n_blocks, 0, stream>>>(
      block_sum, row_id_map, n_tokens, n_blocks);

  // use 128-bit load/store
  constexpr int kFragSize = 16 / sizeof(T);
  // assert(dim % kFragSize == 0);

  // one block per source token
  const int threads = std::min(dim / kFragSize, 1024);
  permute_kernel<T><<<n_tokens, threads, 0, stream>>>(
      tokens, permuted_tokens, row_id_map, n_tokens, n_experts, dim);
}

template <typename T>
void launch_unpermute_kernel(
    const T* permuted_tokens,  // [n_permuted_tokens, dim]
    T* tokens,                 // [n_tokens, dim]
    int* row_id_map,           // [n_experts, n_tokens] => dst row
    const T* probs,            // [n_tokens, n_experts]
    const int n_tokens,
    const int n_experts,
    const int dim,
    cudaStream_t stream) {
  // use 128-bit load/store
  constexpr int kFragSize = 16 / sizeof(T);
  // assert(dim % kFragSize == 0);

  // up to 1024 threads per block
  int threads = std::min(dim / kFragSize, 1024);
  // shared memory for each expert
  size_t smem_bytes = n_experts * sizeof(T);

  // each block corresponds to one token
  unpermute_kernel<T><<<n_tokens, threads, smem_bytes, stream>>>(
      permuted_tokens, tokens, row_id_map, probs, n_tokens, n_experts, dim);
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> permute_with_mask_map(
    torch::Tensor tokens,       // [n_tokens, dim]
    torch::Tensor routing_map,  // [n_tokens, n_experts] bool tensor
    int64_t topk) {
  const auto n_tokens = tokens.size(0);
  const auto dim = tokens.size(1);
  const auto n_experts = routing_map.size(1);

  const auto options = tokens.options();
  const auto int32_options = options.dtype(torch::kInt32);

  // number of tokens per thread to process
  constexpr int BLOCK_SIZE = 128;
  const int64_t n_blocks =
      std::min<int64_t>((n_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE, 1024);
  const auto type = tokens.scalar_type();

  const auto n_permuted_tokens = n_tokens * topk;
  auto permuted_tokens = torch::empty({n_permuted_tokens, dim}, options);
  auto block_sum = torch::empty({n_experts, n_blocks}, int32_options);
  auto row_id_map = torch::empty({n_experts, n_tokens}, int32_options);

  auto* stream = at::cuda::getCurrentCUDAStream().stream();

#define LAUNCH_PERMUTE_KERNEL(DType)                                           \
  launch_permute_kernel<DType, BLOCK_SIZE>(const_data_ptr<DType>(tokens),      \
                                           data_ptr<DType>(permuted_tokens),   \
                                           routing_map.const_data_ptr<bool>(), \
                                           block_sum.data_ptr<int>(),          \
                                           row_id_map.data_ptr<int>(),         \
                                           n_tokens,                           \
                                           n_experts,                          \
                                           topk,                               \
                                           dim,                                \
                                           n_blocks,                           \
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

torch::Tensor unpermute_with_mask_map(
    torch::Tensor permuted_tokens,  // [n_permuted_tokens, dim]
    torch::Tensor row_id_map,       // [n_experts, n_tokens] => dst row
    torch::Tensor probs             // [n_tokens, n_experts]
) {
  const auto dim = permuted_tokens.size(1);
  const auto n_experts = row_id_map.size(0);
  const auto n_tokens = row_id_map.size(1);
  const auto type = permuted_tokens.scalar_type();

  const auto options = permuted_tokens.options();
  // [n_tokens, dim]
  auto tokens = torch::empty({n_tokens, dim}, options);

  auto* stream = at::cuda::getCurrentCUDAStream().stream();

#define LAUNCH_UNPERMUTE_KERNEL(DType)                                   \
  launch_unpermute_kernel<DType>(const_data_ptr<DType>(permuted_tokens), \
                                 data_ptr<DType>(tokens),                \
                                 row_id_map.data_ptr<int>(),             \
                                 const_data_ptr<DType>(probs),           \
                                 n_tokens,                               \
                                 n_experts,                              \
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
