#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include "fused_moe_kernels.h"

namespace llm::kernel {

torch::Tensor apply_fused_moe(torch::Tensor hidden_states,
                              torch::Tensor w1,
                              torch::Tensor w2,
                              torch::Tensor topk_weight,
                              torch::Tensor topk_ids,
                              bool inplace) {
  // Check Constraints
  // CHECK_EQ(hidden_states.shape()[1], w1.shape()[2]);
  return torch::Tensor();
}
}  // namespace llm::kernel