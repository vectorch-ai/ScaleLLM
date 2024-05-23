#pragma once

#include <torch/torch.h>

namespace llm::kernel {
// don't implement the feature of quant temporarily
torch::Tensor apply_fused_moe(torch::Tensor hidden_states,
                              torch::Tensor w13,
                              torch::Tensor w2,
                              torch::Tensor topk_weight,
                              torch::Tensor topk_ids,
                              bool inplace);
}  // namespace llm::kernel
