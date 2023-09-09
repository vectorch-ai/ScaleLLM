#pragma once

#include <torch/torch.h>

#include "attention.h"
#include "feedforward.h"
#include "layers/norm.h"
#include "memory/kv_cache.h"
#include "models/model_args.h"
#include "models/parameters.h"

namespace llm {

class TransformerBlockImpl : public torch::nn::Module {
 public:
  TransformerBlockImpl(const ModelArgs& args,
                       const ParallelArgs& parallel_args,
                       const torch::ScalarType& dtype,
                       const torch::Device& device) {
    // register submodules
    attention_ = register_module("attention",
                                 Attention(args, parallel_args, dtype, device));
    feed_forward_ = register_module(
        "feed_forward", FeedForward(args, parallel_args, dtype, device));
    attention_norm_ = register_module(
        "attention_norm", RMSNorm(args.dim(), args.norm_eps(), dtype, device));
    ffn_norm_ = register_module(
        "ffn_norm", RMSNorm(args.dim(), args.norm_eps(), dtype, device));
  }

  torch::Tensor forward(torch::Tensor x,
                        torch::Tensor positions,
                        KVCache& kv_cache,
                        const InputParameters& input_params) {
    auto h =
        x + attention_(attention_norm_(x), positions, kv_cache, input_params);
    auto out = h + feed_forward_(ffn_norm_(h));
    return out;
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    attention_->load_state_dict(state_dict.select("attention."));
    feed_forward_->load_state_dict(state_dict.select("feed_forward."));
    attention_norm_->load_state_dict(state_dict.select("attention_norm."));
    ffn_norm_->load_state_dict(state_dict.select("ffn_norm."));
  }

 private:
  // parameter members, must be registered
  Attention attention_{nullptr};

  FeedForward feed_forward_{nullptr};

  RMSNorm attention_norm_{nullptr};

  RMSNorm ffn_norm_{nullptr};
};
TORCH_MODULE(TransformerBlock);

}  // namespace llm
