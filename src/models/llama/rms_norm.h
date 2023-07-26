#pragma once

#include <c10/core/TensorImpl.h>
#include <torch/nn/module.h>
#include <torch/torch.h>

#include "models/linear.h"
#include "model_args.h"

namespace llm {

// Root mean square normalization
class RMSNormImpl : public torch::nn::Module {
 public:
  RMSNormImpl(int64_t dim, float eps);

  torch::Tensor forward(torch::Tensor input);

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict);

private:
  torch::Tensor norm(torch::Tensor x);

  // parameter members, must be registered
  torch::Tensor weight_{nullptr};

  // configs
  float eps_;
};
TORCH_MODULE(RMSNorm);

}  // namespace llm
