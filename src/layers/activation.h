#pragma once

#include <torch/torch.h>
namespace llm {
namespace detail {
torch::Tensor gelu(torch::Tensor x);

torch::Tensor gelu_pytorch_tanh(torch::Tensor x);

torch::Tensor gelu_fast(torch::Tensor x);

torch::Tensor gelu_new(torch::Tensor x);

torch::Tensor relu(torch::Tensor x);

torch::Tensor silu(torch::Tensor x);

}  // namespace detail

using ActFunc = torch::Tensor (*)(torch::Tensor);
class Activation {
 public:
  static ActFunc get(const std::string& name, const torch::Device& device);
};

}  // namespace llm
