#include <ATen/cuda/CUDAContext.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <iostream>
#include <unordered_map>

#include "fused_moe_kernels.h"
namespace llm::kernel {

template <typename T>
__global__ void fused_moe_kernel() {}

torch::Tensor apply_fused_moe(torch::Tensor hidden_states,
                              torch::Tensor w13,
                              torch::Tensor w2,
                              torch::Tensor topk_weight,
                              torch::Tensor topk_ids,
                              bool inplace) {
  // Check Constraints
  // match the number of hidden_size
  CHECK(hidden_states.sizes()[1] == w13.sizes()[2]);
  // match topk shape
  CHECK(topk_weight.sizes() == topk_ids.sizes());

  auto M = hidden_states.sizes()[0];  // num_tokens
  auto E = w13.sizes()[0];  // w13 [n_experts,2*intermediate_size,hidden_size]
  auto N = w13.sizes()[1];
  // load kernel config(Now we use the default config)
  std::unordered_map<std::string, int> configs;
  if (M <= E) {
    configs["BLOCK_SIZE_M"] = 16;
    configs["BLOCK_SIZE_N"] = 32;
    configs["BLOCK_SIZE_K"] = 64;
    configs["GROUP_SIZE_M"] = 1;
  } else {
    configs["BLOCK_SIZE_M"] = 64;
    configs["BLOCK_SIZE_N"] = 64;
    configs["BLOCK_SIZE_K"] = 32;
    configs["GROUP_SIZE_M"] = 8;
  }
  // Create intermediate_cache
  auto intermediate_cache1 = torch::empty((M, topk_ids.sizes()[1], N),
                                          hidden_states.device(),
                                          hidden_states.dtype());
  auto intermediate_cache2 = torch::empty((M, topk_ids.sizes()[1], N / 2),
                                          hidden_states.device(),
                                          hidden_states.dtype());
  auto intermediate_cache3 =
      torch::empty((M),
                   hidden_states.device(M, topk_ids.sizes()[1], w2.sizes()[1]),
                   hidden_states.dtype());
  // moe_align_block_size

  return torch::Tensor();
}
}  // namespace llm::kernel