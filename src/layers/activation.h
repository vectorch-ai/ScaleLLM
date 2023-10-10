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

// fused with multiplication to calculate act(x) * y
torch::Tensor gelu_with_mul(torch::Tensor x);

torch::Tensor gelu_pytorch_tanh_with_mul(torch::Tensor x);

torch::Tensor gelu_fast_with_mul(torch::Tensor x);

torch::Tensor gelu_new_with_mul(torch::Tensor x);

torch::Tensor relu_with_mul(torch::Tensor x);

torch::Tensor silu_with_mul(torch::Tensor x);

}  // namespace detail

using ActFunc = torch::Tensor (*)(torch::Tensor);
class Activation {
 public:
  static ActFunc get_act_func(const std::string& name,
                              const torch::Device& device);

  // fused with multiplication
  // calculate act(x) * y where x = input[0] and y = input[1]
  static ActFunc get_act_with_mul_func(const std::string& name,
                                       const torch::Device& device);
};

}  // namespace llm
