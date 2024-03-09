#include "flash_attn_handler.h"

#include <torch/torch.h>

#include "kernels/flash_attn/flash_api.h"
#include "memory/kv_cache.h"
#include "models/input_parameters.h"

namespace llm {

FlashAttnHandler::FlashAttnHandler(float scale,
                                   torch::optional<torch::Tensor> alibi_slopes)
    : scale_(scale), alibi_slopes_(alibi_slopes) {}

// batch prefill for attention, optimized for prefill stage
void FlashAttnHandler::batch_prefill(
    const torch::Tensor& query,  // [n_tokens, n_heads, head_dim]
    const torch::Tensor& key,    // [n_tokens, n_kv_heads, head_dim]
    const torch::Tensor& value,  // [n_tokens, n_kv_heads, head_dim]
    KVCache& kv_cache,           // where to store and retrieval key and value
    const InputParameters& input_params,  // input paras used for attention
    torch::Tensor& output) {
  // append key and value to kv_cache
  // TODO: use a seperate steam since we don't need to wait for the result
  kv_cache.set_kv_cache(input_params.new_cache_slots, key, value);

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
                 /*softmax_scale=*/scale_,
                 /*window_size_left=*/-1,
                 /*window_size_right=*/0,
                 /*num_splits=*/1);
}

// batch decode for attention, optimized for decode stage
// support multiple queries: one sequence with multiple query tokens
void FlashAttnHandler::batch_decode(
    const torch::Tensor& query,  // [n_tokens, n_heads, head_dim]
    const torch::Tensor& key,    // [n_tokens, n_kv_heads, head_dim]
    const torch::Tensor& value,  // [n_tokens, n_kv_heads, head_dim]
    KVCache& kv_cache,           // where to store and retrieval key and value
    const InputParameters& input_params,  // input paras used for attention
    torch::Tensor& output) {
  // append key and value to kv_cache
  kv_cache.set_kv_cache(input_params.new_cache_slots, key, value);

  // TODO: enable split once the core dump issue fixed
  const int32_t num_splits = 1;
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
                 scale_,
                 /*window_size_left=*/-1,
                 /*window_size_right=*/0,
                 num_splits);
}

}  // namespace llm
