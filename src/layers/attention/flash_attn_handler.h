#pragma once

#include <torch/torch.h>

#include "attention_handler.h"
#include "memory/kv_cache.h"
#include "models/input_parameters.h"

namespace llm {

// an flash attn implementation for attention operations
class FlashAttnHandler : public AttentionHandler {
 public:
  FlashAttnHandler(float scale, torch::optional<torch::Tensor> alibi_slopes);

  virtual ~FlashAttnHandler() = default;

  // set workspace for temporary storage before calling any attention operations
  void set_workspace(const torch::Tensor& workspace) override {}

  // batch prefill for attention, optimized for prefill stage
  void batch_prefill(
      const torch::Tensor& query,  // [n_tokens, n_heads, head_dim]
      const torch::Tensor& key,    // [n_tokens, n_kv_heads, head_dim]
      const torch::Tensor& value,  // [n_tokens, n_kv_heads, head_dim]
      KVCache& kv_cache,           // where to store and retrieval key and value
      const InputParameters& input_params,  // input paras used for attention
      torch::Tensor& output) override;

  // batch decode for attention, optimized for decode stage
  // support multiple queries: one sequence with multiple query tokens
  void batch_decode(
      const torch::Tensor& query,  // [n_tokens, n_heads, head_dim]
      const torch::Tensor& key,    // [n_tokens, n_kv_heads, head_dim]
      const torch::Tensor& value,  // [n_tokens, n_kv_heads, head_dim]
      KVCache& kv_cache,           // where to store and retrieval key and value
      const InputParameters& input_params,  // input paras used for attention
      torch::Tensor& output) override;

  // expose this function for testing
  void batch_prefill(
      const torch::Tensor& query,           // [n_tokens, n_heads, head_dim]
      const torch::Tensor& key,             // [n_tokens, n_kv_heads, head_dim]
      const torch::Tensor& value,           // [n_tokens, n_kv_heads, head_dim]
      const torch::Tensor& q_cu_seq_lens,   // [n_seqs + 1]
      const torch::Tensor& kv_cu_seq_lens,  // [n_seqs + 1]
      torch::Tensor& output);

 private:
  // scale factor
  float scale_ = 0.0;

  // alibi slops
  torch::optional<torch::Tensor> alibi_slopes_;
};

}  // namespace llm
