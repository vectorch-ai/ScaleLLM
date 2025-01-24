#pragma once

#include <torch/torch.h>

#include "handler.h"
#include "memory/kv_cache.h"
#include "models/parameters.h"

namespace llm {

// an flash attn implementation for attention operations
class ScaleAttnHandler : public AttentionHandler {
 public:
  // create a flash attn handler with rope positional embedding
  ScaleAttnHandler(float scale,
                    int64_t rotary_dim,
                    int64_t max_position,
                    float rope_scaling,
                    float rope_theta,
                    bool interleaved,
                    const torch::TensorOptions& options);

  // constructor for attention with alibi
  ScaleAttnHandler(float scale, std::optional<torch::Tensor> alibi_slopes);

  virtual ~ScaleAttnHandler() = default;

  std::tuple<torch::Tensor, torch::Tensor> apply_pos_emb(
      const torch::Tensor& query,
      const torch::Tensor& key,
      const torch::Tensor& /*positions*/) override {
    // no positional embedding since we will apply pos emb on the fly
    return {query, key};
  }

  // batch decode for attention, optimized for decode stage
  // support multiple queries: one sequence with multiple query tokens
  void batch_decode(
      const torch::Tensor& query,  // [n_tokens, n_heads, head_dim]
      const KVCache& kv_cache,     // where to store and retrieval key and value
      const InputParameters& input_params,  // input paras used for attention
      int32_t sliding_window,               // sliding window size
      torch::Tensor& output) override;

  // append key and value to kv_cache
  void append_kv_cache(
      KVCache& kv_cache,           // where to store and retrieval key and value
      const torch::Tensor& key,    // [n_tokens, n_kv_heads, head_dim]
      const torch::Tensor& value,  // [n_tokens, n_kv_heads, head_dim]
      const InputParameters& input_params) override;

 private:
  // scale factor
  float scale_ = 0.0;

  // alibi slops
  std::optional<torch::Tensor> alibi_slopes_;
};

}  // namespace llm
