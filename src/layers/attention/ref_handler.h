#pragma once

#include <torch/torch.h>

#include "handler.h"
#include "layers/pos_embedding.h"
#include "memory/kv_cache.h"
#include "models/parameters.h"

namespace llm {

// an pytorch implementation handler for attention operations, used for testing
class RefHandler : public AttentionHandler {
 public:
  // create a flash attn handler with rope positional embedding
  RefHandler(float sm_scale,
             float logits_soft_cap,
             int64_t rotary_dim,
             int64_t max_position,
             torch::Tensor inv_freq,
             bool interleaved,
             const torch::TensorOptions& options);

  // create a flash attn handler with alibi slopes
  RefHandler(float sm_scale,
             float logits_soft_cap,
             torch::optional<torch::Tensor> alibi_slopes);

  virtual ~RefHandler() = default;

  // set workspace for temporary storage before calling any attention operations
  void set_workspace(const torch::Tensor& workspace) override {}

  // apply positional embedding to query and key if needed
  std::tuple<torch::Tensor, torch::Tensor> apply_pos_emb(
      const torch::Tensor& query,
      const torch::Tensor& key,
      const torch::Tensor& positions) override;

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
  // softmax scale factor
  float sm_scale_ = 0.0;

  // logits softcap
  float logits_soft_cap_ = 0.0;

  // ROPE positional embedding
  RotaryEmbedding pos_emb_{nullptr};

  // alibi slops
  torch::optional<torch::Tensor> alibi_slopes_;
};

}  // namespace llm
