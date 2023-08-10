#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

namespace llm {

// Rotary positional embedding
class RotaryPositionalEmbeddingImpl : public torch::nn::Module {
 public:
  RotaryPositionalEmbeddingImpl(int64_t rotary_dim, int64_t seq_len);

  // inplace rotary positional embedding
  void forward(torch::Tensor& query,    // [num_tokens, n_heads, head_dim]
               torch::Tensor& key,      // [num_tokens, n_kv_heads, head_dim]
               torch::Tensor positions  // [num_tokens]
  ) const;

 private:
  torch::Tensor freqs_cis_;
};
TORCH_MODULE(RotaryPositionalEmbedding);

}  // namespace llm
