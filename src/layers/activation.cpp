#include "activation.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <boost/algorithm/string.hpp>
#include <cmath>
#include <kernels/activation_kernels.h>

namespace llm {
namespace detail {
torch::Tensor gelu(torch::Tensor x) {
  namespace F = torch::nn::functional;
  return F::gelu(x);
}

torch::Tensor gelu_pytorch_tanh(torch::Tensor x) {
  namespace F = torch::nn::functional;
  return F::gelu(x, F::GELUFuncOptions().approximate("tanh"));
}

torch::Tensor gelu_fast(torch::Tensor x) {
  return 0.5 * x *
         (1.0 +
          torch::tanh(0.7978845608028654f * x * (1.0 + 0.044715f * x * x)));
}

torch::Tensor gelu_new(torch::Tensor x) {
  return 0.5 * x *
         (1.0 + torch::tanh(0.7978845608028654f *
                            (x + 0.044715f * torch::pow(x, 3.0))));
}

torch::Tensor relu(torch::Tensor x) {
  namespace F = torch::nn::functional;
  return F::relu(x);
}

torch::Tensor silu(torch::Tensor x) {
  namespace F = torch::nn::functional;
  return F::silu(x);
}
}  // namespace detail

ActFunc Activation::get(const std::string& name, const torch::Device& device) {
  CHECK(!name.empty()) << "Activation function name cannot be empty";
  using namespace detail;
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
