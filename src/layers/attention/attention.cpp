#include "attention.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/torch.h>

#include "kernels/flash_attn/flash_api.h"

DEFINE_bool(disable_custom_kernels, false, "disable all custom kernels");

namespace llm {
AttentionImpl::AttentionImpl(int64_t n_heads,
                             int64_t n_kv_heads,
                             int64_t head_dim,
                             float scale,
                             torch::ScalarType /*dtype*/,
                             const torch::Device& device)
    : n_heads_(n_heads),
      n_kv_heads_(n_kv_heads),
      head_dim_(head_dim),
      scale_(scale) {
  CHECK(n_heads % n_kv_heads == 0)
      << "n_heads " << n_heads << " not divisible by n_kv_heads " << n_kv_heads;
}

torch::Tensor AttentionImpl::forward(const torch::Tensor& query,
                                     const torch::Tensor& key,
                                     const torch::Tensor& value,
                                     KVCache& kv_cache,
                                     const InputParameters& input_params) {
  const int64_t n_tokens = query.size(0);
  // (n_tokens, n_heads, head_dim)
  auto q = query.view({n_tokens, n_heads_, head_dim_});
  auto k = key.view({n_tokens, n_kv_heads_, head_dim_});
  auto v = value.view({n_tokens, n_kv_heads_, head_dim_});

  auto output = torch::empty_like(q);
  if (kv_cache.empty()) {
    // empty kv_cache, it is a batch for memory profiling
    detail::varlen_masked_self_attention(q,
                                         k,
                                         v,
                                         input_params.q_cu_seq_lens,
                                         /*alibi_slopes=*/torch::nullopt,
                                         input_params.q_max_seq_len,
                                         scale_,
                                         output);
  } else {
    // store k/v into cache based on slots
    kv_cache.set_kv_cache(input_params.new_cache_slots, k, v);
    detail::multiple_query_masked_self_attention(
        q,
        kv_cache,
        input_params.q_cu_seq_lens,
        input_params.kv_cu_seq_lens,
        input_params.block_tables,
        /*alibi_slopes=*/torch::nullopt,
        input_params.q_max_seq_len,
        input_params.kv_max_seq_len,
        scale_,
        output);
  }

  // reshape output to [n_tokens, n_heads * head_dim]
  return output.view({n_tokens, -1});
}

namespace detail {

void varlen_masked_self_attention(
    const torch::Tensor& query,        // [n_tokens, n_heads, head_dim]
    const torch::Tensor& key,          // [n_tokens, n_kv_heads, head_dim]
    const torch::Tensor& value,        // [n_tokens, n_kv_heads, head_dim]
    const torch::Tensor& cu_seq_lens,  // [n_seqs + 1]
    torch::optional<torch::Tensor> alibi_slopes,  // [n_heads]
    int32_t max_seq_len,                          // maximum sequence length
    float scale,                                  // scale for softmax
    torch::Tensor& output) {
  if (query.is_cuda() && !FLAGS_disable_custom_kernels) {
    // use cuda kernel
    return varlen_masked_self_attention_cuda(query,
                                             key,
                                             value,
                                             cu_seq_lens,
                                             alibi_slopes,
                                             max_seq_len,
                                             scale,
                                             output);
  }
  CHECK(false)
      << "multiple_query_masked_self_attention not implemented for CPU";
}

void multiple_query_masked_self_attention(
    const torch::Tensor& query,           // [n_q_tokens, n_heads, head_dim]
    const KVCache& kv_cache,              // where to get key and value
    const torch::Tensor& q_cu_seq_lens,   // [n_seqs + 1]
    const torch::Tensor& kv_cu_seq_lens,  // [n_seqs + 1]
    const torch::optional<torch::Tensor>
        block_tables,  // [n_seqs, max_n_blocks]
    const torch::optional<torch::Tensor> alibi_slopes,  // [n_heads]
    int32_t q_max_seq_len,  // maximum sequence length for Q
    int32_t k_max_seq_len,  // maximum sequence length for K/V
    float scale,
    torch::Tensor& output) {
  if (query.is_cuda() && !FLAGS_disable_custom_kernels) {
    // use cuda kernel
    // TODO: enable split once the core dump issue fixed
    const int32_t num_splits = 1;
    auto [key_cache, value_cache] = kv_cache.get_kv_cache();
    return multiple_query_masked_self_attention_cuda(query,
                                                     key_cache,
                                                     value_cache,
                                                     q_cu_seq_lens,
                                                     kv_cu_seq_lens,
                                                     block_tables,
                                                     alibi_slopes,
                                                     q_max_seq_len,
                                                     k_max_seq_len,
                                                     scale,
                                                     output,
                                                     num_splits);
  }
  CHECK(false)
      << "multiple_query_masked_self_attention not implemented for CPU";
}

void varlen_masked_self_attention_cuda(
    const torch::Tensor& query,        // [n_tokens, n_heads, head_dim]
    const torch::Tensor& key,          // [n_tokens, n_kv_heads, head_dim]
    const torch::Tensor& value,        // [n_tokens, n_kv_heads, head_dim]
    const torch::Tensor& cu_seq_lens,  // [n_seqs + 1]
    torch::optional<torch::Tensor> alibi_slopes,  // [n_heads]
    int32_t max_seq_len,                          // maximum sequence length
    float scale,                                  // scale for softmax
    torch::Tensor& output) {
  mha_varlen_fwd(output,
                 query,
                 key,
                 value,
                 cu_seq_lens,
                 cu_seq_lens,
                 /*block_table=*/torch::nullopt,
                 alibi_slopes,
                 max_seq_len,
                 max_seq_len,
                 /*softmax_scale=*/scale,
                 /*window_size_left=*/-1,
                 /*window_size_right=*/0,
                 /*num_splits=*/1);
}

void multiple_query_masked_self_attention_cuda(
    const torch::Tensor& query,          // [n_q_tokens, n_heads, head_dim]
    const torch::Tensor& key,            // [..., n_kv_heads, head_dim]
    const torch::Tensor& value,          // [..., n_kv_heads, head_dim]
    const torch::Tensor& q_cu_seq_lens,  // [n_seqs + 1]
    const torch::Tensor& k_cu_seq_lens,  // [n_seqs + 1]
    const torch::optional<torch::Tensor>
        block_tables,  // [n_seqs, max_n_blocks]
    const torch::optional<torch::Tensor> alibi_slopes,  // [n_heads]
    int32_t q_max_seq_len,  // maximum sequence length for Q
    int32_t k_max_seq_len,  // maximum sequence length for K/V
    float scale,
    torch::Tensor& output,
    int32_t num_splits) {
  mha_varlen_fwd(output,
                 query,
                 key,
                 value,
                 q_cu_seq_lens,
                 k_cu_seq_lens,
                 block_tables,
                 alibi_slopes,
                 q_max_seq_len,
                 k_max_seq_len,
                 /*softmax_scale=*/scale,
                 /*window_size_left=*/-1,
                 /*window_size_right=*/0,
                 num_splits);
}

}  // namespace detail

}  // namespace llm
