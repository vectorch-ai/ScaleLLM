#include "transformer.h"

#include <torch/torch.h>

#include <memory>

#include "attention.h"
#include "layers/embedding.h"
#include "layers/linear.h"
#include "models/model_args.h"
#include "transformer_block.h"

namespace llm {

TransformerImpl::TransformerImpl(const ModelArgs& args, int64_t world_size) {
  // register submodules
  tok_embeddings_ = register_module(
      "tok_embeddings",
      ParallelEmbedding(args.vocab_size(), args.dim(), world_size));
  blocks_ = register_module("layers", torch::nn::ModuleList());
  layers_.reserve(args.n_layers());
  for (int i = 0; i < args.n_layers(); i++) {
    auto block = TransformerBlock(i, args, world_size);
    layers_.push_back(block);
    blocks_->push_back(block);
  }
  norm_ = register_module("norm", RMSNorm(args.dim(), args.norm_eps()));
  output_ = register_module(
      "output",
      ColumnParallelLinear(args.dim(), args.vocab_size(), world_size));
}

torch::Tensor TransformerImpl::forward(
    torch::Tensor tokens,
    torch::Tensor positions,
    const std::vector<int64_t>& cu_seq_lens) {
  auto h = tok_embeddings_->forward(tokens);
  for (auto layer : layers_) {
    h = layer->forward(h, positions, cu_seq_lens);
  }
  h = norm_->forward(h);
  auto output = output_->forward(h).to(torch::kFloat32);
  return output;
}

// load the weight from the checkpoint
void TransformerImpl::load_state_dict(const StateDict& state_dict) {
  tok_embeddings_->load_state_dict(state_dict.select("tok_embeddings."));
  // call each layer's load_state_dict function
  for (int i = 0; i < layers_.size(); i++) {
    layers_[i]->load_state_dict(
        state_dict.select("layers." + std::to_string(i) + "."));
  }
  norm_->load_state_dict(state_dict.select("norm."));
  output_->load_state_dict(state_dict.select("output."));
}

}  // namespace llm
