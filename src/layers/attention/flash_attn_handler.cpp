#include "flash_attn_handler.h"

#include <cuda_runtime.h>
#include <torch/torch.h>

#include "kernels/flash_attn/flash_api.h"
#include "memory/kv_cache.h"
#include "models/parameters.h"

namespace llm {

FlashAttnHandler::FlashAttnHandler(float scale,
                                   int64_t rotary_dim,
                                   int64_t max_position,
                                   torch::Tensor inv_freq,
                                   bool interleaved,
                                   const torch::TensorOptions& options)
    : sm_scale_(scale) {
  // register rotary positional embedding
  pos_emb_ =
      RotaryEmbedding(rotary_dim, max_position, inv_freq, interleaved, options);
}

FlashAttnHandler::FlashAttnHandler(float scale,
                                   torch::optional<torch::Tensor> alibi_slopes)
    : sm_scale_(scale), alibi_slopes_(alibi_slopes) {}

FlashAttnHandler::~FlashAttnHandler() {}

std::tuple<torch::Tensor, torch::Tensor> FlashAttnHandler::apply_pos_emb(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& positions) {
  // for alibi scenarios, the pos_emb_ is not defined
  if (positions.defined() && pos_emb_) {
    return pos_emb_(query, key, positions);
  }
  return {query, key};
}

// batch prefill for attention, optimized for prefill stage
void FlashAttnHandler::batch_prefill(
    const torch::Tensor& query,           // [n_tokens, n_heads, head_dim]
    const torch::Tensor& key,             // [n_tokens, n_kv_heads, head_dim]
    const torch::Tensor& value,           // [n_tokens, n_kv_heads, head_dim]
    const InputParameters& input_params,  // input paras used for attention
    int32_t sliding_window,               // sliding window size
    torch::Tensor& output) {
  // don't use kv cache in prefill stage
  mha_varlen_fwd(output,
                 query,
                 key,
                 value,
                 input_params.q_cu_seq_lens,
                 input_params.kv_cu_seq_lens,
                 /*block_table=*/torch::nullopt,
                 alibi_slopes_,
                 input_params.q_max_seq_len,
                 input_params.kv_max_seq_len,
                 /*softmax_scale=*/sm_scale_,
                 /*window_size_left=*/sliding_window,
                 /*window_size_right=*/0,
                 /*num_splits=*/0);
}

// batch decode for attention, optimized for decode stage
// support multiple queries: one sequence with multiple query tokens
void FlashAttnHandler::batch_decode(
    const torch::Tensor& query,           // [n_tokens, n_heads, head_dim]
    const KVCache& kv_cache,              // where to retrieval key and value
    const InputParameters& input_params,  // input paras used for attention
    int32_t sliding_window,               // sliding window size
    torch::Tensor& output) {
  auto [key_cache, value_cache] = kv_cache.get_kv_cache();
  mha_varlen_fwd(output,
                 query,
                 key_cache,
                 value_cache,
                 input_params.q_cu_seq_lens,
                 input_params.kv_cu_seq_lens,
                 input_params.block_tables,
                 alibi_slopes_,
                 input_params.q_max_seq_len,
                 input_params.kv_max_seq_len,
                 sm_scale_,
                 /*window_size_left=*/sliding_window,
                 /*window_size_right=*/0,
                 /*num_splits=*/0);
}

// append key and value to kv_cache
void FlashAttnHandler::append_kv_cache(
    KVCache& kv_cache,           // where to store key and value
    const torch::Tensor& key,    // [n_tokens, n_kv_heads, head_dim]
    const torch::Tensor& value,  // [n_tokens, n_kv_heads, head_dim]
    const InputParameters& input_params) {
  // append key and value to kv_cache
  if (!kv_cache.empty()) {
    kv_cache.set_kv_cache(input_params.new_cache_slots, key, value);
  }
}

}  // namespace llm
