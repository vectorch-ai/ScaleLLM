#pragma once

#include <torch/nn/module.h>
#include <torch/torch.h>

#include "model_args.h"
#include "models/layers.h"

namespace llm {

class AttentionImpl : public torch::nn::Module {
 public:
  AttentionImpl(const ModelArgs& args, int64_t world_size);

  torch::Tensor forward(torch::Tensor x,
                        int64_t start_pos,
                        torch::Tensor freqs_cis,
                        torch::Tensor mask);

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict);

  // parameter members, must be registered
  ColumnParallelLinear wq_{nullptr};

  ColumnParallelLinear wk_{nullptr};

  ColumnParallelLinear wv_{nullptr};

  RowParallelLinear wo_{nullptr};

  // state variable members
  torch::Tensor cache_k_{nullptr};
  torch::Tensor cache_v_{nullptr};

  // configs
  int64_t world_size_;
};

TORCH_MODULE(Attention);

}  // namespace llm
