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
  kv_cache.set_kv_cache(input_params.new_cache_slots, k, v);

  auto output = torch::empty_like(q);
  detail::multiple_query_masked_self_attention(q,
                                               kv_cache,
                                               input_params.q_cu_seq_lens,
                                               input_params.kv_cu_seq_lens,
                                               input_params.block_tables,
                                               alibi_slopes_,
                                               input_params.q_max_seq_len,
                                               input_params.kv_max_seq_len,
                                               scale_,
                                               output);
  return output.view({num_tokens, -1});
}

}  // namespace llm
