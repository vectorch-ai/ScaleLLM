#pragma once

#include <torch/torch.h>

#include "attention.h"
#include "feedforward.h"
#include "layers/norm.h"
#include "models/input_parameters.h"
#include "models/model_args.h"
#include "memory/kv_cache.h"

namespace llm {

class TransformerBlockImpl : public torch::nn::Module {
 public:
  TransformerBlockImpl(int32_t layer_id,
                       const ModelArgs& args,
                       int64_t world_size,
                       const torch::Device& device)
      : world_size_(world_size) {
    // register submodules
    attention_ = register_module("attention", Attention(args, world_size, device));
    feed_forward_ = register_module("feed_forward",
                                    FeedForward(/*dim=*/args.dim(),
                                                /*hidden_dim=*/4 * args.dim(),
                                                args.multiple_of(),
                                                args.ffn_dim_multiplier(),
                                                world_size,
                                                device));
    attention_norm_ =
        register_module("attention_norm", RMSNorm(args.dim(), args.norm_eps(), device));
    ffn_norm_ =
        register_module("ffn_norm", RMSNorm(args.dim(), args.norm_eps(), device));
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

  // configs
  int64_t world_size_;
};
TORCH_MODULE(TransformerBlock);

}  // namespace llm
