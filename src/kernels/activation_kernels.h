#pragma once
#include <torch/torch.h>

namespace llm::kernel {

torch::Tensor gelu_new(torch::Tensor input);
torch::Tensor gelu_fast(torch::Tensor input);
torch::Tensor silu(torch::Tensor input);

// fused with multiplication
// calculate act(x) * y where x = input[0] and y = input[1]
torch::Tensor gelu_new_with_mul(torch::Tensor input);
torch::Tensor gelu_fast_with_mul(torch::Tensor input);
torch::Tensor silu_with_mul(torch::Tensor input);

}  // namespace llm::kernel
