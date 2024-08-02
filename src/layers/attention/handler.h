#pragma once

#include <c10/core/TensorOptions.h>
#include <torch/torch.h>

#include <memory>

#include "memory/kv_cache.h"
#include "models/model_args.h"
#include "models/parameters.h"

namespace llm {

// a handler for attention operations
class AttentionHandler {
 public:
  virtual ~AttentionHandler() = default;

  // -1 means not needed
  virtual int64_t get_estimate_workspace_size() { return -1; };

  // set workspace for temporary storage before calling any attention operations
  virtual void set_workspace(const torch::Tensor& workspace) {}

  // apply positional embedding to query and key if needed
  virtual std::tuple<torch::Tensor, torch::Tensor> apply_pos_emb(
      const torch::Tensor& query,
      const torch::Tensor& key,
      const torch::Tensor& /*positions*/) = 0;

  // batch prefill for attention, optimized for prefill stage
  // common optimizations include: 1> leverage tensor-core 2> contuguous memory
  // limitation?: all sequences in the batch are all in prefill stage
  virtual void batch_prefill(
      const torch::Tensor& query,           // [n_tokens, n_heads, head_dim]
      const torch::Tensor& key,             // [n_tokens, n_kv_heads, head_dim]
      const torch::Tensor& value,           // [n_tokens, n_kv_heads, head_dim]
      const InputParameters& input_params,  // input paras used for attention
      int32_t sliding_window,               // sliding window size
      torch::Tensor& output) = 0;

  // batch decode for attention, optimized for decode stage
  // support multiple queries: one sequence with multiple query tokens
  virtual void batch_decode(
      const torch::Tensor& query,           // [n_tokens, n_heads, head_dim]
      const KVCache& kv_cache,              // where to retrieval key and value
      const InputParameters& input_params,  // input paras used for attention
      int32_t sliding_window,               // sliding window size
      torch::Tensor& output) = 0;

  // append key and value to kv_cache
  virtual void append_kv_cache(
      KVCache& kv_cache,           // where to store and retrieval key and value
      const torch::Tensor& key,    // [n_tokens, n_kv_heads, head_dim]
      const torch::Tensor& value,  // [n_tokens, n_kv_heads, head_dim]
      const InputParameters& input_params) = 0;

  // create an attention handler
  static std::unique_ptr<AttentionHandler> create_handler(
      const ModelArgs& args,
      const torch::TensorOptions& options) {
    return create_handler_with_alibi(args, torch::nullopt, options);
  }

  // create an attention handler with alibi slopes
  static std::unique_ptr<AttentionHandler> create_handler_with_alibi(
      const ModelArgs& args,
      torch::optional<torch::Tensor> alibi_slopes,
      const torch::TensorOptions& options);

  // create an attention handler with ROPE
  static std::unique_ptr<AttentionHandler> create_handler_with_rope(
      const ModelArgs& args,
      bool interleaved,
      const torch::TensorOptions& options);
};

}  // namespace llm
