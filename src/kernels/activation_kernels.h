#pragma once
#include <torch/torch.h>

namespace llm::kernel {

torch::Tensor gelu_new(torch::Tensor input);
torch::Tensor gelu_fast(torch::Tensor input);
torch::Tensor silu(torch::Tensor input);

}  // namespace llm::kernel
