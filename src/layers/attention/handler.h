#pragma once

#include <gflags/gflags.h>
#include <torch/torch.h>

#include <memory>

#include "memory/kv_cache.h"
#include "models/input_parameters.h"
#include "models/model_args.h"

namespace llm {

// an handler for attention operations
class AttentionHandler {
 public:
  virtual ~AttentionHandler() = default;

  // -1 means not needed
  virtual int64_t get_estimate_workspace_size() { return -1; };

  // set workspace for temporary storage before calling any attention operations
  virtual void set_workspace(const torch::Tensor& workspace) {}

  // batch prefill for attention, optimized for prefill stage
  // common optimizations include: 1> leverage tensor-core 2> contuguous memory
  // limitation?: all sequences in the batch are all in prefill stage
  virtual void batch_prefill(
      const torch::Tensor& query,           // [n_tokens, n_heads, head_dim]
      const torch::Tensor& key,             // [n_tokens, n_kv_heads, head_dim]
      const torch::Tensor& value,           // [n_tokens, n_kv_heads, head_dim]
      const InputParameters& input_params,  // input paras used for attention
      torch::Tensor& output) = 0;

  // batch decode for attention, optimized for decode stage
  // support multiple queries: one sequence with multiple query tokens
  virtual void batch_decode(
      const torch::Tensor& query,           // [n_tokens, n_heads, head_dim]
      const KVCache& kv_cache,              // where to retrieval key and value
      const InputParameters& input_params,  // input paras used for attention
      torch::Tensor& output) = 0;

  // append key and value to kv_cache
  virtual void append_kv_cache(
      KVCache& kv_cache,           // where to store and retrieval key and value
      const torch::Tensor& key,    // [n_tokens, n_kv_heads, head_dim]
      const torch::Tensor& value,  // [n_tokens, n_kv_heads, head_dim]
      const InputParameters& input_params) = 0;

  // create an attention handler
  static std::unique_ptr<AttentionHandler> create(
      const ModelArgs& args,
      const torch::Device& device,
      torch::optional<torch::Tensor> alibi_slopes = torch::nullopt);
};

}  // namespace llm
