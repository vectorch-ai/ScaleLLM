#pragma once

#include <torch/nn/module.h>
#include <torch/torch.h>

#include <optional>

#include "model_args.h"
#include "models/layers.h"

namespace llm {

class FeedForwardImpl : public torch::nn::Module {
 public:
  FeedForwardImpl(int64_t dim,
                  int64_t hidden_dim,
                  int64_t multiple_of,
                  std::optional<float> ffn_dim_multiplier,
                  int64_t world_size);

  torch::Tensor forward(torch::Tensor x);

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict);

 private:
  // parameter members, must be registered
  ColumnParallelLinear w1_{nullptr};
  RowParallelLinear w2_{nullptr};
  ColumnParallelLinear w3_{nullptr};
};
TORCH_MODULE(FeedForward);

}  // namespace llm
