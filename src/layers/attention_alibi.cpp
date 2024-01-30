#include "attention_alibi.h"

#include <gflags/gflags.h>
#include <torch/torch.h>

#include "attention.h"
#include "common/logging.h"

namespace llm {

AttentionWithAlibiImpl::AttentionWithAlibiImpl(int64_t n_heads,
                                               int64_t n_kv_heads,
                                               int64_t head_dim,
                                               float scale,
                                               torch::Tensor alibi_slopes,
                                               torch::ScalarType /*dtype*/,
                                               const torch::Device& device)
    : n_heads_(n_heads),
      n_kv_heads_(n_kv_heads),
      head_dim_(head_dim),
      scale_(scale),
      alibi_slopes_(alibi_slopes.to(device)) {
  GCHECK(n_heads % n_kv_heads == 0)
      << "n_heads " << n_heads << " not divisible by n_kv_heads " << n_kv_heads;
  GCHECK(alibi_slopes.dim() == 1 && alibi_slopes.size(0) == n_heads)
      << "alibi_slopes should be a 1D tensor of size " << n_heads << " but got "
      << alibi_slopes_.sizes() << " instead.";
}

torch::Tensor AttentionWithAlibiImpl::forward(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    KVCache& kv_cache,
    const InputParameters& input_params) {
  const int64_t num_tokens = query.size(0);

  // (num_tokens, n_heads, head_dim)
  auto q = query.view({num_tokens, n_heads_, head_dim_});
  auto k = key.view({num_tokens, n_kv_heads_, head_dim_});
  auto v = value.view({num_tokens, n_kv_heads_, head_dim_});

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
                                         alibi_slopes_,
                                         input_params.max_seq_len,
                                         scale_,
                                         sliced_output);
  }

  if (num_prompt_tokens < num_tokens) {
    // process sequences without prompt tokens (decode)
    auto sliced_output = output.slice(/*dim=*/0, /*start=*/num_prompt_tokens);
    auto sliced_query = q.slice(/*dim=*/0, /*start=*/num_prompt_tokens);
    detail::single_query_masked_self_attention(kv_cache,
                                               static_cast<int32_t>(n_kv_heads_),
                                               sliced_query,
                                               input_params.block_tables,
                                               input_params.context_lens,
                                               alibi_slopes_,
                                               input_params.max_context_len,
                                               scale_,
                                               sliced_output);
  }
  return output.view({num_tokens, -1});
}

}  // namespace llm
