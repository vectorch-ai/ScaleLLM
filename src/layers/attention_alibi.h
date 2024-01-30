#pragma once

#include <torch/torch.h>

#include "layers/pos_embedding.h"
#include "memory/kv_cache.h"
#include "models/input_parameters.h"

namespace llm {

// Attention with Alibi relative positional embedding
class AttentionWithAlibiImpl : public torch::nn::Module {
 public:
  AttentionWithAlibiImpl(int64_t n_heads,
                         int64_t n_kv_heads,
                         int64_t head_dim,
                         float scale,
                         torch::Tensor alibi_slopes,  // [n_heads]
                         torch::ScalarType dtype,
                         const torch::Device& device);

  // query: [num_tokens, n_heads, head_dim]
  // key/value: [num_tokens, n_kv_heads, head_dim]
  // return: [num_tokens, n_heads, head_dim]
  torch::Tensor forward(const torch::Tensor& query,
                        const torch::Tensor& key,
                        const torch::Tensor& value,
                        KVCache& kv_cache,
                        const InputParameters& input_params);

 private:
  RotaryEmbedding pos_emb_{nullptr};

  int64_t n_heads_ = 0;
  int64_t n_kv_heads_ = 0;
  int64_t head_dim_ = 0;

  // scale factor
  float scale_ = 0.0;

  // alibi slopes for each head [n_heads]
  torch::Tensor alibi_slopes_;
};
TORCH_MODULE(AttentionWithAlibi);

}  // namespace llm
