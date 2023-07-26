#include "transformer_block.h"

#include <torch/nn/module.h>
#include <torch/torch.h>

#include "attention.h"
#include "feedforward.h"
#include "model_args.h"
#include "rms_norm.h"

namespace llm {

TransformerBlockImpl::TransformerBlockImpl(const ModelArgs& args,
                                           int32_t layer_id,
                                           int64_t world_size): world_size_(world_size) {
  // register submodules
  attention_ = register_module("attention", Attention(args, world_size));
  feed_forward_ = register_module("feed_forward",
                                  FeedForward(args.dim(),
                                              4 * args.dim(),
                                              args.multiple_of(),
                                              args.ffn_dim_multiplier(),
                                              world_size));
  attention_norm_ =
      register_module("attention_norm", RMSNorm(args.dim(), args.norm_eps()));
  ffn_norm_ = register_module("ffn_norm", RMSNorm(args.dim(), args.norm_eps()));
}

torch::Tensor TransformerBlockImpl::forward(torch::Tensor input) {
  return input;
}

// load the weight from the checkpoint
void TransformerBlockImpl::load_state_dict(const StateDict& state_dict) {
  // call each submodule's load_state_dict function
  attention_->load_state_dict(state_dict.select("attention."));
  feed_forward_->load_state_dict(state_dict.select("feed_forward."));
  attention_norm_->load_state_dict(state_dict.select("attention_norm."));
  ffn_norm_->load_state_dict(state_dict.select("ffn_norm."));
}

}  // namespace llm
