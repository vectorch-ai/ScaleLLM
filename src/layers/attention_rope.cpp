#include "attention_rope.h"

#include <gflags/gflags.h>
#include <torch/torch.h>

#include "attention.h"
#include "common/logging.h"

namespace llm {

AttentionWithRoPEImpl::AttentionWithRoPEImpl(int64_t n_heads,
                                             int64_t n_kv_heads,
                                             int64_t head_dim,
                                             float scale,
                                             int64_t rotary_dim,
                                             float rope_sclaing,
                                             float rope_theta,
                                             int64_t max_position,
                                             bool interleaved,
                                             torch::ScalarType dtype,
                                             const torch::Device& device)
    : n_heads_(n_heads),
      n_kv_heads_(n_kv_heads),
      head_dim_(head_dim),
      scale_(scale) {
  GCHECK(n_heads % n_kv_heads == 0)
      << "n_heads " << n_heads << " not divisible by n_kv_heads " << n_kv_heads;
  // register rotary positional embedding
  pos_emb_ = register_module("pos_emb",
                             RotaryEmbedding(rotary_dim,
                                             max_position,
                                             rope_sclaing,
                                             rope_theta,
                                             interleaved,
                                             dtype,
                                             device));

  kv_head_mapping_ = register_buffer(
      "kv_head_mapping",
      detail::prepare_kv_head_mapping(n_heads, n_kv_heads, device));
}

torch::Tensor AttentionWithRoPEImpl::forward(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& positions,
    KVCache& kv_cache,
    const InputParameters& input_params) {
  const int64_t num_tokens = query.size(0);

  // (num_tokens, n_heads, head_dim)
  auto q = query.view({num_tokens, n_heads_, head_dim_});
  auto k = key.view({num_tokens, n_kv_heads_, head_dim_});
  auto v = value.view({num_tokens, n_kv_heads_, head_dim_});

  // (num_tokens, n_local_heads, head_dim)
  // apply positional embedding
  std::tie(q, k) = pos_emb_(q, k, positions);

  // store k/v into cache based on slots
  kv_cache.set_kv_cache(input_params.slot_ids, k, v);

  auto output = torch::empty_like(q);
  const auto num_prompt_tokens = input_params.num_prompt_tokens;
  if (num_prompt_tokens > 0) {
    // process sequences with prompt tokens (prefill)
    auto sliced_output =
        output.slice(/*dim=*/0, /*start=*/0, /*end=*/num_prompt_tokens);
    auto sliced_query =
        q.slice(/*dim=*/0, /*start=*/0, /*end=*/num_prompt_tokens);
    auto sliced_key =
        k.slice(/*dim=*/0, /*start=*/0, /*end=*/num_prompt_tokens);
    auto sliced_value =
        v.slice(/*dim=*/0, /*start=*/0, /*end=*/num_prompt_tokens);
    detail::varlen_masked_self_attention(sliced_query,
                                         sliced_key,
                                         sliced_value,
                                         input_params.cu_seq_lens,
                                         /*alibi_slopes=*/torch::nullopt,
                                         input_params.max_seq_len,
                                         scale_,
                                         sliced_output);
  }

  if (num_prompt_tokens < num_tokens) {
    // process sequences without prompt tokens (decode)
    auto sliced_output = output.slice(/*dim=*/0, /*start=*/num_prompt_tokens);
    auto sliced_query = q.slice(/*dim=*/0, /*start=*/num_prompt_tokens);
    detail::single_query_masked_self_attention(kv_cache,
                                               kv_head_mapping_,
                                               sliced_query,
                                               input_params.block_tables,
                                               input_params.context_lens,
                                               /*alibi_slopes=*/torch::nullopt,
                                               input_params.max_context_len,
                                               scale_,
                                               sliced_output);
  }
  return output.view({num_tokens, -1});
}

}  // namespace llm
