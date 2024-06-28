#include "activation.h"

#include <gflags/gflags_declare.h>
#include <glog/logging.h>
#include <kernels/activation_kernels.h>
#include <torch/torch.h>

#include <boost/algorithm/string.hpp>

DECLARE_bool(disable_custom_kernels);

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

torch::Tensor gelu_with_mul(torch::Tensor x) {
  namespace F = torch::nn::functional;
  const auto chunked = x.chunk(/*chunks=*/2, /*dim=*/-1);
  return F::gelu(chunked[0]) * chunked[1];
}

torch::Tensor gelu_pytorch_tanh_with_mul(torch::Tensor x) {
  const auto chunked = x.chunk(/*chunks=*/2, /*dim=*/-1);
  return gelu_pytorch_tanh(chunked[0]) * chunked[1];
}

torch::Tensor gelu_fast_with_mul(torch::Tensor x) {
  const auto chunked = x.chunk(/*chunks=*/2, /*dim=*/-1);
  return gelu_fast(chunked[0]) * chunked[1];
}

torch::Tensor gelu_new_with_mul(torch::Tensor x) {
  const auto chunked = x.chunk(/*chunks=*/2, /*dim=*/-1);
  return gelu_new(chunked[0]) * chunked[1];
}

torch::Tensor relu_with_mul(torch::Tensor x) {
  namespace F = torch::nn::functional;
  const auto chunked = x.chunk(/*chunks=*/2, /*dim=*/-1);
  return F::relu(chunked[0]) * chunked[1];
}

torch::Tensor silu_with_mul(torch::Tensor x) {
  namespace F = torch::nn::functional;
  const auto chunked = x.chunk(/*chunks=*/2, /*dim=*/-1);
  return F::silu(chunked[0]) * chunked[1];
}
}  // namespace detail

ActFunc Activation::get_act_func(const std::string& name,
                                 const torch::Device& device) {
  CHECK(!name.empty()) << "Activation function name cannot be empty";
  using namespace detail;
  if (boost::iequals(name, "gelu")) {
    return gelu;
  }
  // TODO: need to support quick_gelu
  if (boost::iequals(name, "quick_gelu")) {
    return gelu;
  }
  if (boost::iequals(name, "gelu_fast")) {
    return device.is_cuda() && !FLAGS_disable_custom_kernels ? kernel::gelu_fast
                                                             : gelu_fast;
  }
  if (boost::iequals(name, "gelu_new")) {
    return device.is_cuda() && !FLAGS_disable_custom_kernels ? kernel::gelu_new
                                                             : gelu_new;
  }
  if (boost::iequals(name, "gelu_pytorch_tanh")) {
    return gelu_pytorch_tanh;
  }
  if (boost::iequals(name, "relu")) {
    return relu;
  }
  if (boost::iequals(name, "silu")) {
    return device.is_cuda() && !FLAGS_disable_custom_kernels ? kernel::silu
                                                             : silu;
  }

  LOG(ERROR) << "Unsupported activation function: " << name;
  return nullptr;
}

ActFunc Activation::get_act_with_mul_func(const std::string& name,
                                          const torch::Device& device) {
  CHECK(!name.empty()) << "Activation function name cannot be empty";
  using namespace detail;
  if (boost::iequals(name, "gelu")) {
    return gelu_with_mul;
  }
  if (boost::iequals(name, "gelu_fast")) {
    return device.is_cuda() && !FLAGS_disable_custom_kernels
               ? kernel::gelu_fast_with_mul
               : gelu_fast_with_mul;
  }
  if (boost::iequals(name, "gelu_new")) {
    return device.is_cuda() && !FLAGS_disable_custom_kernels
               ? kernel::gelu_new_with_mul
               : gelu_new_with_mul;
  }
  if (boost::iequals(name, "gelu_pytorch_tanh")) {
    return gelu_pytorch_tanh_with_mul;
  }
  if (boost::iequals(name, "relu")) {
    return relu_with_mul;
  }
  if (boost::iequals(name, "silu")) {
    return device.is_cuda() && !FLAGS_disable_custom_kernels
               ? kernel::silu_with_mul
               : silu_with_mul;
  }

  LOG(ERROR) << "Unsupported activation function: " << name;
  return nullptr;
}

}  // namespace llm
