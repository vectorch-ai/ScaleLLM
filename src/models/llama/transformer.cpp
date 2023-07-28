#include "transformer.h"

#include <torch/nn/module.h>
#include <torch/torch.h>

#include <memory>

#include "attention.h"
#include "model_args.h"
#include "models/layers.h"
#include "rms_norm.h"
#include "transformer_block.h"

namespace llm {

static torch::Tensor precompute_freqs_cis(int64_t dim,
                                          int64_t max_seq_len,
                                          float theta = 10000.0f) {
  auto range = torch::arange(0, dim, 2);
  auto slice =
      range.slice(/*dim=*/0, /*start=*/0, /*end=*/dim / 2).to(torch::kFloat32);
  auto freqs = 1.0 / torch::pow(theta, slice / dim);
  auto t = torch::arange(0, max_seq_len, 1).to(torch::kFloat32);
  freqs = torch::outer(t, freqs).to(torch::kFloat32);
  return torch::polar(torch::ones_like(freqs), freqs);
}

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

  // calculate freqs_cis
  freqs_cis_ =
      precompute_freqs_cis(args.dim() / args.n_heads(), args.max_seq_len() * 2);
}

torch::Tensor TransformerImpl::forward(torch::Tensor tokens,
                                       int64_t start_pos) {
  constexpr float negative_infinity = -std::numeric_limits<float>::infinity();

  const auto batch_size = tokens.size(/*dim=*/0);
  const auto seq_len = tokens.size(/*dim=*/1);
  auto h = tok_embeddings_->forward(tokens);
  auto freqs_cis = freqs_cis_.slice(/*dim=*/0, start_pos, start_pos + seq_len);
  torch::Tensor mask;
  if (seq_len > 1) {
    mask = torch::full({1, 1, seq_len, seq_len}, negative_infinity);
    mask = torch::triu(mask, /*diagonal=*/start_pos + 1).type_as(h);
  }
  for (auto layer : layers_) {
    h = layer->forward(h, start_pos, freqs_cis, mask);
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
