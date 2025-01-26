#include "scale_attn_handler.h"

#include <torch/torch.h>

#include "kernels/attention/attn_api.h"
#include "memory/kv_cache.h"
#include "models/parameters.h"

namespace llm {

ScaleAttnHandler::ScaleAttnHandler(float sm_scale,
                                   float logits_soft_cap,
                                   int64_t rotary_dim,
                                   int64_t max_position,
                                   torch::Tensor inv_freq,
                                   bool interleaved,
                                   const torch::TensorOptions& options)
    : sm_scale_(sm_scale), logits_soft_cap_(logits_soft_cap) {
  // register rotary positional embedding
  pos_emb_ =
      RotaryEmbedding(rotary_dim, max_position, inv_freq, interleaved, options);
}

ScaleAttnHandler::ScaleAttnHandler(float sm_scale,
                                   float logits_soft_cap,
                                   torch::optional<torch::Tensor> alibi_slopes)
    : sm_scale_(sm_scale),
      logits_soft_cap_(logits_soft_cap),
      alibi_slopes_(alibi_slopes) {}

std::tuple<torch::Tensor, torch::Tensor> ScaleAttnHandler::apply_pos_emb(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& positions) {
  // for alibi scenarios, the pos_emb_ is not defined
  if (positions.defined() && pos_emb_) {
    return pos_emb_(query, key, positions);
  }
  return {query, key};
}

// batch decode for attention, optimized for decode stage
// support multiple queries: one sequence with multiple query tokens
void ScaleAttnHandler::batch_decode(
    const torch::Tensor& query,           // [n_tokens, n_heads, head_dim]
    const KVCache& kv_cache,              // where to retrieval key and value
    const InputParameters& input_params,  // input paras used for attention
    int32_t sliding_window,               // sliding window size
    torch::Tensor& output) {
  auto [key_cache, value_cache] = kv_cache.get_kv_cache();
  const auto block_size = kv_cache.block_size();

  paged_kv_varlen_mha(output,
                      query,
                      key_cache,
                      value_cache,
                      input_params.q_cu_seq_lens,
                      input_params.kv_cu_seq_lens,
                      input_params.block_tables,
                      input_params.cu_block_lens,
                      alibi_slopes_,
                      block_size,
                      input_params.q_max_seq_len,
                      input_params.kv_max_seq_len,
                      sm_scale_,
                      logits_soft_cap_,
                      sliding_window);
}

// append key and value to kv_cache
void ScaleAttnHandler::append_kv_cache(
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
