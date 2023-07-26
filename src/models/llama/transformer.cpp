#include "transformer.h"

#include <torch/nn/module.h>
#include <torch/torch.h>
#include <memory>

#include "attention.h"
#include "model_args.h"
#include "models/embedding.h"
#include "models/linear.h"
#include "transformer_block.h"
#include "rms_norm.h"

namespace llm {

TransformerImpl::TransformerImpl(const ModelArgs& args, int64_t world_size) {
  // register submodules
  tok_embeddings_ = register_module(
      "tok_embeddings",
      ColumnParallelEmbedding(args.vocab_size(), args.dim(), world_size));
  layers_ = register_module("layers", torch::nn::ModuleList());
  for (int i = 0; i < args.n_layers(); i++) {
    auto block = TransformerBlock(args, i, world_size);
    layers_->push_back(block);
  }
  norm_ = register_module("norm", RMSNorm(args.dim(), args.norm_eps()));
  output = register_module(
      "output",
      ColumnParallelLinear(args.dim(), args.vocab_size(), world_size));
}

torch::Tensor TransformerImpl::forward(torch::Tensor input) { return input; }

// load the weight from the checkpoint
void TransformerImpl::load_state_dict(const StateDict& state_dict) {
  tok_embeddings_->load_state_dict(state_dict.select("tok_embeddings."));
  // call each layer's load_state_dict function
  for (int i = 0; i < layers_->size(); i++) {
    auto block = std::dynamic_pointer_cast<TransformerBlockImpl>(layers_[i]);
    block->load_state_dict(
        state_dict.select("layers." + std::to_string(i) + "."));
  }
  norm_->load_state_dict(state_dict.select("norm."));
  output->load_state_dict(state_dict.select("output."));
}

}  // namespace llm
