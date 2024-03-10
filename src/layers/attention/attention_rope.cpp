#include "attention_rope.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include "attention.h"

namespace llm {

AttentionWithRoPEImpl::AttentionWithRoPEImpl(int64_t n_heads,
                                             int64_t n_kv_heads,
                                             int64_t head_dim,
                                             int64_t rotary_dim,
                                             float rope_sclaing,
                                             float rope_theta,
                                             int64_t max_position,
                                             bool interleaved,
                                             torch::ScalarType dtype,
                                             const torch::Device& device,
                                             AttentionHandler* handler)
    : n_heads_(n_heads),
      n_kv_heads_(n_kv_heads),
      head_dim_(head_dim),
      handler_(handler) {
  CHECK(handler_ != nullptr);
  CHECK(n_heads % n_kv_heads == 0)
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

  // append key and value to kv_cache
  handler_->append_kv_cache(kv_cache, k, v, input_params);

  auto output = torch::empty_like(q);
  if (input_params.all_prefill_sequences) {
    handler_->batch_prefill(q, k, v, input_params, output);
  } else {
    handler_->batch_decode(q, kv_cache, input_params, output);
  }

  return output.view({num_tokens, -1});
}

}  // namespace llm
