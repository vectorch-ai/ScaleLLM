#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

namespace llm {

// Rotary positional embedding
class RotaryPositionalEmbeddingImpl : public torch::nn::Module {
 public:
  RotaryPositionalEmbeddingImpl(int64_t rotary_dim, int64_t seq_len);

  void forward(
      torch::Tensor& query,  // [batch_size, seq_len, n_heads, head_dim]
      torch::Tensor& key,    // [batch_size, seq_len, n_heads, head_dim]
      int64_t start_pos,
      int64_t seq_len) const;

 private:
  torch::Tensor freqs_cis_;
};
TORCH_MODULE(RotaryPositionalEmbedding);

}  // namespace llm
