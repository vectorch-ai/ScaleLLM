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

void launch_permute_row_id_kernel(
    const bool* topk_ids,  // [n_tokens, topk] bool
    int* sorted_ids,       // [n_permuted_tokens] dst_idx -> t_idx
    int* expert_ids,       // [n_permuted_tokens + n_experts] b_idx -> e_idx
    const int n_tokens,
    const int n_experts,
    const int64_t block_size,
    cudaStream_t stream) {
  // // step1: transpose routing_map to [n_experts, n_tokens] and calculate
  // block
  // // sum for each expert
  // block_sum_kernel<BLOCK_SIZE><<<n_experts, n_blocks, 0, stream>>>(
  //     routing_map, row_id_map, block_sum, n_tokens, n_experts, n_blocks);

  // // step2: calculate index in permuted tokens for each token
  // // launch_row_id_kernel;
  // row_id_map_kernel<BLOCK_SIZE><<<n_experts, n_blocks, 0, stream>>>(
  //     block_sum, row_id_map, n_tokens, n_blocks);
}

}  // namespace

// returns sorted_row_ids, expert_ids
std::tuple<torch::Tensor, torch::Tensor> permute_row_id_map(
    torch::Tensor topk_ids,  // [n_tokens, topk]
    int64_t n_experts,
    int64_t block_size) {
  const auto n_tokens = topk_ids.size(0);
  const auto topk = topk_ids.size(1);

  const auto n_permuted_tokens = n_tokens * topk;
  const auto options = topk_ids.options();
  const auto int32_options = options.dtype(torch::kInt32);
  auto sorted_indices = torch::zeros(n_permuted_tokens, int32_options);
  auto row_id = torch::range(0, n_permuted_tokens - 1, 1, int32_options);

  // expert ids for each block
  auto expert_ids =
      torch::empty({n_permuted_tokens + n_experts}, int32_options);
  auto sorted_row_ids = torch::empty(n_permuted_tokens, int32_options);

  // number of tokens per thread to process

  auto* stream = at::cuda::getCurrentCUDAStream().stream();

  launch_permute_row_id_kernel(topk_ids.const_data_ptr<bool>(),
                               sorted_row_ids.data_ptr<int>(),
                               expert_ids.data_ptr<int>(),
                               n_tokens,
                               n_experts,
                               block_size,
                               stream);

  return {sorted_row_ids, expert_ids};
}

}  // namespace llm::kernel::moe
