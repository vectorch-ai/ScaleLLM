#pragma once

#include <c10/core/TensorImpl.h>
#include <torch/nn/module.h>
#include <torch/torch.h>

#include "model_args.h"
#include "models/layers.h"

namespace llm {

// Root mean square normalization
class RMSNormImpl : public torch::nn::Module {
 public:
  RMSNormImpl(int64_t dim, float eps = 1e-6);

  torch::Tensor forward(torch::Tensor input);

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict);

  void pretty_print(std::ostream& stream) const override {
    stream << name() << " " << weight_.sizes() << " " << weight_.device();
  }

 private:
  torch::Tensor norm(torch::Tensor x);

  // parameter members, must be registered
  torch::Tensor weight_{nullptr};

  // configs
  float eps_;
};
TORCH_MODULE(RMSNorm);

}  // namespace llm
