#pragma once

#include <torch/nn/module.h>
#include <torch/torch.h>

#include "model_args.h"
#include "attention.h"
#include "rms_norm.h"
#include "models/embedding.h"
#include "models/linear.h"

namespace llm {

class TransformerImpl : public torch::nn::Module {
 public:
  TransformerImpl(const ModelArgs& args, int64_t world_size);

  torch::Tensor forward(torch::Tensor input);

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict);

private:
  // parameter members, must be registered
  ColumnParallelEmbedding tok_embeddings_{nullptr};

  torch::nn::ModuleList layers_{nullptr};

  RMSNorm norm_{nullptr};

  ColumnParallelLinear output{nullptr};

  // configs
  int64_t world_size_;
};
TORCH_MODULE(Transformer);

}  // namespace llm
