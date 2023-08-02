#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include "torch_utils/state_dict.h"

namespace llm {

// Root mean square normalization
class RMSNormImpl : public torch::nn::Module {
 public:
  RMSNormImpl(int64_t dim, float eps = 1e-6) : eps_(eps) {
    weight_ = register_parameter(
        "weight", torch::empty({dim}), /*requires_grad=*/false);
  }

  torch::Tensor forward(torch::Tensor input) {
    auto output = norm(input.to(torch::kFloat)).type_as(input);
    return output * weight_;
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    const auto weight = state_dict.get_tensor("weight");
    if (weight.defined()) {
      CHECK_EQ(weight_.sizes(), weight.sizes()) << "weight size mismatch";
      weight_.copy_(weight);
    } else {
      LOG(WARNING) << "weight is not defined";
    }
  }

  void pretty_print(std::ostream& stream) const override {
    stream << name() << " " << weight_.sizes() << " " << weight_.device();
  }

 private:
  torch::Tensor norm(const torch::Tensor& x) {
    return x * torch::rsqrt(x.pow(2).mean(-1, /*keepdim*/ true) + eps_);
  }

  // parameter members, must be registered
  torch::Tensor weight_{nullptr};

  // configs
  float eps_;
};
TORCH_MODULE(RMSNorm);

}  // namespace llm
