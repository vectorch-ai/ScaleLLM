#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include "model_loader/state_dict.h"

namespace llm {

// Root mean square normalization
class RMSNormImpl : public torch::nn::Module {
 public:
  RMSNormImpl(int64_t dim,
              float eps,
              const torch::ScalarType& dtype,
              const torch::Device& device)
      : eps_(eps) {
    weight_ = register_parameter(
        "weight",
        torch::empty({dim}, torch::dtype(dtype).device(device)),
        /*requires_grad=*/false);
  }

  torch::Tensor forward(torch::Tensor input) {
    // TODO: do we really need to cast to float?
    auto output = norm(input.to(torch::kFloat)).type_as(input);
    return output * weight_;
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    const auto weight = state_dict.get_tensor("weight");
    if (weight.defined()) {
      CHECK_EQ(weight_.sizes(), weight.sizes())
          << "weight size mismatch for " << name();
      weight_.copy_(weight);
      is_loaded_ = true;
    }
  }

  // whether the weight is loaded
  void verify_loaded_weights(const std::string& prefix = "") const {
    CHECK(is_loaded_) << "weight is not loaded for " << prefix + ".weight";
  }

  void pretty_print(std::ostream& stream) const override {
    stream << name() << " " << weight_.sizes() << " " << weight_.device();
  }

 private:
  torch::Tensor norm(const torch::Tensor& x) {
    return x *
           torch::rsqrt(
               x.pow(/*exponent=*/2).mean(/*dim=*/-1, /*keepdim=*/true) + eps_);
  }

  // parameter members, must be registered
  torch::Tensor weight_{nullptr};

  // whether the weight is loaded
  bool is_loaded_ = false;

  // configs
  float eps_;
};
TORCH_MODULE(RMSNorm);

}  // namespace llm
