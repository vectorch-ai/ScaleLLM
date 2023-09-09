#pragma once

#include <c10/core/ScalarType.h>
#include <torch/torch.h>

#include "attention.h"
#include "layers/embedding.h"
#include "layers/linear.h"
#include "layers/norm.h"
#include "memory/kv_cache.h"
#include "models/model_args.h"
#include "models/parallel_args.h"
#include "models/parameters.h"
#include "transformer_block.h"

// port LLAMA's model to C++ API:
// https://github.com/facebookresearch/llama/blob/main/llama/model.py
namespace llm {

class TransformerImpl : public torch::nn::Module {
 public:
  TransformerImpl(const ModelArgs& args,
                  const ParallelArgs& parallel_args,
                  const torch::ScalarType& dtype,
                  const torch::Device& device) {
    // register submodules
    tok_embeddings_ = register_module(
        "tok_embeddings",
        ParallelEmbedding(
            args.vocab_size(), args.dim(), parallel_args, dtype, device));
    blocks_ = register_module("layers", torch::nn::ModuleList());
    layers_.reserve(args.n_layers());
    for (int i = 0; i < args.n_layers(); i++) {
      auto block = TransformerBlock(args, parallel_args, dtype, device);
      layers_.push_back(block);
      blocks_->push_back(block);
    }
    norm_ = register_module(
        "norm", RMSNorm(args.dim(), args.norm_eps(), dtype, device));
    output_ = register_module(
        "output",
        ColumnParallelLinear(
            args.dim(), args.vocab_size(), parallel_args, dtype, device));
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  torch::Tensor forward(torch::Tensor tokens,
                        torch::Tensor positions,
                        std::vector<KVCache>& kv_caches,
                        const InputParameters& input_params) {
    auto h = tok_embeddings_(tokens);
    for (size_t i = 0; i < layers_.size(); i++) {
      auto& layer = layers_[i];
      h = layer(h, positions, kv_caches[i], input_params);
    }
    h = norm_(h);
    auto output = output_(h).to(torch::kFloat32);
    return output;
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    tok_embeddings_->load_state_dict(state_dict.select("tok_embeddings."));
    // call each layer's load_state_dict function
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->load_state_dict(
          state_dict.select("layers." + std::to_string(i) + "."));
    }
    norm_->load_state_dict(state_dict.select("norm."));
    output_->load_state_dict(state_dict.select("output."));
  }

 private:
  // parameter members, must be registered
  ParallelEmbedding tok_embeddings_{nullptr};

  torch::nn::ModuleList blocks_{nullptr};
  // hold same data but different type as blocks_ to avoid type cast
  std::vector<TransformerBlock> layers_;

  RMSNorm norm_{nullptr};

  ColumnParallelLinear output_{nullptr};
};
TORCH_MODULE(Transformer);

}  // namespace llm
