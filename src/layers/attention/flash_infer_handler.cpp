#include "flash_infer_handler.h"

#include <torch/torch.h>

#include "memory/kv_cache.h"
#include "models/parameters.h"

namespace llm {

FlashInferHandler::FlashInferHandler(float scale,
                                     int64_t rotary_dim,
                                     int64_t max_position,
                                     float rope_scaling,
                                     float rope_theta,
                                     bool interleaved,
                                     const torch::TensorOptions& options) {
  LOG(FATAL) << "Not implemented yet";
}

FlashInferHandler::FlashInferHandler(
    float scale,
    torch::optional<torch::Tensor> alibi_slopes)
    : scale_(scale), alibi_slopes_(alibi_slopes) {}

// batch prefill for attention, optimized for prefill stage
void FlashInferHandler::batch_prefill(
    const torch::Tensor& query,           // [n_tokens, n_heads, head_dim]
    const torch::Tensor& key,             // [n_tokens, n_kv_heads, head_dim]
    const torch::Tensor& value,           // [n_tokens, n_kv_heads, head_dim]
    const InputParameters& input_params,  // input paras used for attention
    int32_t sliding_window,               // sliding window size
    torch::Tensor& output) {
  // TODO: add implementation
  LOG(FATAL) << "Not implemented yet";
}

// batch decode for attention, optimized for decode stage
// support multiple queries: one sequence with multiple query tokens
void FlashInferHandler::batch_decode(
    const torch::Tensor& query,           // [n_tokens, n_heads, head_dim]
    const KVCache& kv_cache,              // where to retrieval key and value
    const InputParameters& input_params,  // input paras used for attention
    int32_t sliding_window,               // sliding window size
    torch::Tensor& output) {
  // TODO: add implementation
  LOG(FATAL) << "Not implemented yet";
}

// append key and value to kv_cache
void FlashInferHandler::append_kv_cache(
    KVCache& kv_cache,           // where to store key and value
    const torch::Tensor& key,    // [n_tokens, n_kv_heads, head_dim]
    const torch::Tensor& value,  // [n_tokens, n_kv_heads, head_dim]
    const InputParameters& input_params) {
  // TODO: add implementation
  LOG(FATAL) << "Not implemented yet";
}

}  // namespace llm
