#pragma once

#include <c10/core/TensorImpl.h>
#include <glog/logging.h>
#include <torch/torch.h>

namespace llm {

// Self attention
class SelfAttentionImpl : public torch::nn::Module {
 public:
  torch::Tensor forward(
      torch::Tensor query,  // [batch_size, seq_len, n_heads, head_dim]
      torch::Tensor key,    // [batch_size, seq_len, n_heads, head_dim]
      torch::Tensor value,  // [batch_size, seq_len, n_heads, head_dim]
      torch::Tensor mask,   // [batch_size, seq_len, seq_len]
      float scale) const;
};
TORCH_MODULE(SelfAttention);

}  // namespace llm
