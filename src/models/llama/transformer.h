#pragma once

#include <torch/torch.h>

#include "attention.h"
#include "layers/embedding.h"
#include "layers/linear.h"
#include "layers/norm.h"
#include "models/model_args.h"
#include "transformer_block.h"

// port LLAMA's model to C++ API:
// https://github.com/facebookresearch/llama/blob/main/llama/model.py
namespace llm {

class TransformerImpl : public torch::nn::Module {
 public:
  TransformerImpl(const ModelArgs& args, int64_t world_size);

  torch::Tensor forward(torch::Tensor tokens, int64_t start_pos);

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict);

 private:
  // parameter members, must be registered
  ParallelEmbedding tok_embeddings_{nullptr};

  torch::nn::ModuleList blocks_{nullptr};
  // hold same data but different type as blocks_ to avoid type cast
  std::vector<TransformerBlock> layers_;

  RMSNorm norm_{nullptr};

  ColumnParallelLinear output_{nullptr};

  // configs
  int64_t world_size_;
};
TORCH_MODULE(Transformer);

}  // namespace llm
