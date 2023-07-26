#pragma once

#include <torch/nn/functional/embedding.h>
#include <torch/nn/module.h>
#include <torch/torch.h>

#include "common/state_dict.h"

namespace llm {

// Linear layer with column parallelism.
// The linear layer is defined as Y = XA + b. A is parallelized along
// its second dimension as A = [A_1, ..., A_p].
class ColumnParallelLinearImpl : public torch::nn::Module {
 public:
  ColumnParallelLinearImpl(int64_t in_features,
                           int64_t out_features,
                           int64_t world_size);

  torch::Tensor forward(torch::Tensor input);

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict);

 private:
  // parameter members, must be registered
  torch::Tensor weight_{nullptr};

  // configs
  int64_t world_size_;
};
TORCH_MODULE(ColumnParallelLinear);

// Linear layer with row parallelism.
//     The linear layer is defined as Y = XA + b. A is parallelized along
//     its first dimension and X along its second dimension as:
//                -   -
//               | A_1 |
//               | .   |
//           A = | .   |       X = [X_1, ..., X_p]
//               | .   |
//               | A_p |
//                -   -
class RowParallelLinearImpl : public torch::nn::Module {
 public:
  RowParallelLinearImpl(int64_t in_features,
                        int64_t out_features,
                        int64_t world_size);

  torch::Tensor forward(torch::Tensor input);

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict);

 private:
  // parameter members, must be registered
  torch::Tensor weight_{nullptr};

  // configs
  int64_t world_size_;
};
TORCH_MODULE(RowParallelLinear);

}  // namespace llm
