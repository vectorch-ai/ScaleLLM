#pragma once

#include <c10/core/TensorImpl.h>
#include <torch/nn/module.h>
#include <torch/torch.h>

#include "attention.h"
#include "feedforward.h"
#include "model_args.h"
#include "rms_norm.h"

namespace llm {

class TransformerBlockImpl : public torch::nn::Module {
 public:
  TransformerBlockImpl(int32_t layer_id,
                       const ModelArgs& args,
                       int64_t world_size);

  torch::Tensor forward(torch::Tensor x,
                        int64_t start_pos,
                        torch::Tensor freqs_cis,
                        torch::Tensor mask);

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict);

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
