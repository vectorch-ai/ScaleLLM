#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include "models/model_args.h"

namespace llm {

// Rotary positional embedding
class RotaryPositionalEmbeddingImpl : public torch::nn::Module {
 public:
  RotaryPositionalEmbeddingImpl(const ModelArgs& args);

  void forward(torch::Tensor& query,
               torch::Tensor& key,
               int64_t start_pos,
               int64_t seq_len) const;

 private:
  torch::Tensor freqs_cis_;
};
TORCH_MODULE(RotaryPositionalEmbedding);

}  // namespace llm
