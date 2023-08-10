#pragma once

#include <torch/torch.h>

#include "attention.h"
#include "feedforward.h"
#include "layers/norm.h"
#include "models/model_args.h"

namespace llm {

class TransformerBlockImpl : public torch::nn::Module {
 public:
  TransformerBlockImpl(int32_t layer_id,
                       const ModelArgs& args,
                       int64_t world_size)
      : world_size_(world_size) {
    // register submodules
    attention_ = register_module("attention", Attention(args, world_size));
    feed_forward_ = register_module("feed_forward",
                                    FeedForward(/*dim=*/args.dim(),
                                                /*hidden_dim=*/4 * args.dim(),
                                                args.multiple_of(),
                                                args.ffn_dim_multiplier(),
                                                world_size));
    attention_norm_ =
        register_module("attention_norm", RMSNorm(args.dim(), args.norm_eps()));
    ffn_norm_ =
        register_module("ffn_norm", RMSNorm(args.dim(), args.norm_eps()));
  }

  torch::Tensor forward(torch::Tensor x,
                        torch::Tensor positions,
                        const std::vector<int64_t>& cu_seq_lens) {
    auto h = x + attention_->forward(
                     attention_norm_->forward(x), positions, cu_seq_lens);
    auto out = h + feed_forward_->forward(ffn_norm_->forward(h));
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
