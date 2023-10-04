#include "activation.h"

#include <torch/torch.h>

#include <boost/algorithm/string.hpp>
#include <cmath>
namespace llm {
namespace {
inline torch::Tensor gelu(torch::Tensor x) {
  namespace F = torch::nn::functional;
  return F::gelu(x);
}

inline torch::Tensor gelu_pytorch_tanh(torch::Tensor x) {
  namespace F = torch::nn::functional;
  return F::gelu(x, F::GELUFuncOptions().approximate("tanh"));
}

inline torch::Tensor gelu_fast(torch::Tensor x) {
  return 0.5 * x *
         (1.0 + torch::tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)));
}

inline torch::Tensor gelu_new(torch::Tensor x) {
  static double sqrt_2_over_pi = std::sqrt(2.0 / M_PI);
  return 0.5 * x *
         (1.0 +
          torch::tanh(sqrt_2_over_pi * (x + 0.044715 * torch::pow(x, 3.0))));
}

inline torch::Tensor relu(torch::Tensor x) {
  namespace F = torch::nn::functional;
  return F::relu(x);
}

inline torch::Tensor silu(torch::Tensor x) {
  namespace F = torch::nn::functional;
  return F::silu(x);
}
}  // namespace

ActFunc Activation::get(const std::string& name) {
  CHECK(!name.empty()) << "Activation function name cannot be empty";

  if (boost::iequals(name, "gelu")) {
    return gelu;
  }
  if (boost::iequals(name, "gelu_fast")) {
    return gelu_fast;
  }
  if (boost::iequals(name, "gelu_new")) {
    return gelu_new;
  }
  if (boost::iequals(name, "gelu_pytorch_tanh")) {
    return gelu_pytorch_tanh;
  }
  if (boost::iequals(name, "relu")) {
    return relu;
  }
  if (boost::iequals(name, "silu")) {
    return silu;
  }

  LOG(ERROR) << "Unsupported activation function: " << name;
  return nullptr;
}

}  // namespace llm
