#pragma once

#include <torch/torch.h>

#include "layers/pos_embedding.h"
#include "memory/kv_cache.h"
#include "models/input_parameters.h"
#include "handler.h"

namespace llm {

// Attention with rotary embedding
// only support linear scaling type for now.
// TODO: add dynamic and yarn scaling type.
class AttentionWithRoPEImpl : public torch::nn::Module {
 public:
  AttentionWithRoPEImpl(int64_t n_heads,
                        int64_t n_kv_heads,
                        int64_t head_dim,
                        int64_t rotary_dim,
                        float rope_sclaing,
                        float rope_theta,
                        int64_t max_position,
                        bool interleaved,
                        torch::ScalarType dtype,
                        const torch::Device& device,
                        AttentionHandler* handler);

  // query: [num_tokens, n_heads, head_dim]
  // key/value: [num_tokens, n_kv_heads, head_dim]
  // return: [num_tokens, n_heads, head_dim]
  torch::Tensor forward(const torch::Tensor& query,
                        const torch::Tensor& key,
                        const torch::Tensor& value,
                        const torch::Tensor& positions,
                        KVCache& kv_cache,
                        const InputParameters& input_params);

 private:
  RotaryEmbedding pos_emb_{nullptr};

  int64_t n_heads_ = 0;
  int64_t n_kv_heads_ = 0;
  int64_t head_dim_ = 0;

  AttentionHandler* handler_ = nullptr;
};
TORCH_MODULE(AttentionWithRoPE);

}  // namespace llm
