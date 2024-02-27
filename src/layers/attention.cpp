#include "attention.h"

#include <c10/util/Optional.h>
#include <gflags/gflags.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/torch.h>

#include "common/logging.h"
#include "kernels/flash_attn/flash_api.h"

DEFINE_bool(disable_custom_kernels, false, "disable all custom kernels");

DEFINE_bool(
    force_use_paged_attention_v2,
    false,
    "force to use paged attention v2 for single_query_masked_self_attention");

// ref to paged_attention_v1 in third_party/vllm
extern void paged_attention_v1(
    torch::Tensor& out,
    torch::Tensor& query,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    int num_kv_heads,
    float scale,
    torch::Tensor& block_tables,
    torch::Tensor& context_lens,
    int block_size,
    int max_context_len,
    const c10::optional<torch::Tensor>& alibi_slopes);

extern void paged_attention_v2(
    torch::Tensor& out,
    torch::Tensor& exp_sums,
    torch::Tensor& max_logits,
    torch::Tensor& tmp_out,
    torch::Tensor& query,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    int num_kv_heads,
    float scale,
    torch::Tensor& block_tables,
    torch::Tensor& context_lens,
    int block_size,
    int max_context_len,
    const c10::optional<torch::Tensor>& alibi_slopes);

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
  GCHECK(n_heads % n_kv_heads == 0)
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

  if (num_prompt_tokens < n_tokens) {
    // process sequences without prompt tokens (decode)
    auto sliced_output = output.slice(/*dim=*/0, /*start=*/num_prompt_tokens);
    auto sliced_query = q.slice(/*dim=*/0, /*start=*/num_prompt_tokens);
    detail::single_query_masked_self_attention(
        kv_cache,
        static_cast<int32_t>(n_kv_heads_),
        sliced_query,
        input_params.block_tables,
        input_params.context_lens,
        /*alibi_slopes=*/torch::nullopt,
        input_params.max_context_len,
        scale_,
        sliced_output);
  }
  return output.view({n_tokens, -1});
}

namespace detail {
using torch::indexing::Slice;
constexpr float negative_infinity = -std::numeric_limits<float>::infinity();

torch::Tensor masked_self_attention(
    const torch::Tensor& query,  // [q_seq_len, n_heads, head_dim]
    const torch::Tensor& key,    // [k_seq_len, n_heads, head_dim]
    const torch::Tensor& value,  // [k_seq_len, n_heads, head_dim]
    const torch::Tensor& mask,   // [n_heads, q_seq_len, k_seq_len]
    float scale) {
  // => [n_heads, q_seq_len, k_seq_len]
  auto scores = torch::einsum("qhd,khd->hqk", {query * scale, key});
  scores = scores.to(torch::kFloat);
  if (mask.defined()) {
    scores += mask;
  }
  scores = torch::softmax(scores, /*dim=*/-1).type_as(query);
  // => [q_seq_len, n_heads, head_dim]
  return torch::einsum("hqk,khd->qhd", {scores, value});
}

// returns IntTensor [n_heads], mapping from query head to kv head
torch::Tensor prepare_kv_head_mapping(int64_t n_heads,
                                      int64_t n_kv_heads,
                                      const torch::Device& device) {
  // prepare kv_head_mapping
  auto kv_head_mapping =
      torch::arange(0, n_kv_heads, torch::dtype(torch::kInt).device(device));
  const auto num_group = n_heads / n_kv_heads;
  if (num_group > 1) {
    kv_head_mapping = kv_head_mapping.repeat_interleave(/*repeats=*/num_group);
  }
  return kv_head_mapping;
}

void varlen_masked_self_attention(
    const torch::Tensor& query,        // [n_tokens, n_heads, head_dim]
    const torch::Tensor& key,          // [n_tokens, n_kv_heads, head_dim]
    const torch::Tensor& value,        // [n_tokens, n_kv_heads, head_dim]
    const torch::Tensor& cu_seq_lens,  // [n_seq + 1]
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
  return varlen_masked_self_attention_generic(
      query, key, value, cu_seq_lens, alibi_slopes, scale, output);
}

void single_query_masked_self_attention(
    const KVCache& kv_cache,  // where to get key and value
    int32_t n_kv_heads,
    const torch::Tensor& query,         // [n_tokens/n_seq, n_heads, head_dim]
    const torch::Tensor& block_tables,  // [n_tokens, num_blocks]
    const torch::Tensor& context_lens,  // [n_tokens]
    torch::optional<torch::Tensor> alibi_slopes,  // [n_heads]
    int32_t max_context_len,                      // maximum context length
    float scale,                                  // scale for softmax
    torch::Tensor& output) {
  if (query.is_cuda() && !FLAGS_disable_custom_kernels) {
    // use cuda kernel
    return single_query_masked_self_attention_cuda(kv_cache,
                                                   n_kv_heads,
                                                   query,
                                                   block_tables,
                                                   context_lens,
                                                   alibi_slopes,
                                                   max_context_len,
                                                   scale,
                                                   output);
  }
  return single_query_masked_self_attention_generic(kv_cache,
                                                    query,
                                                    block_tables,
                                                    context_lens,
                                                    alibi_slopes,
                                                    max_context_len,
                                                    scale,
                                                    output);
}

void varlen_masked_self_attention_generic(
    const torch::Tensor& query,        // [n_tokens, n_heads, head_dim]
    const torch::Tensor& key,          // [n_tokens, n_kv_heads, head_dim]
    const torch::Tensor& value,        // [n_tokens, n_kv_heads, head_dim]
    const torch::Tensor& cu_seq_lens,  // [n_seq + 1]
    torch::optional<torch::Tensor> alibi_slopes,  // [n_heads]
    float scale,                                  // scale for softmax
    torch::Tensor& output) {
  DCHECK(query.size(0) == key.size(0));
  DCHECK(query.size(0) == value.size(0));

  const auto head_dim = query.size(-1);
  torch::Tensor cu_seq_lens_cpu = cu_seq_lens.cpu();
  const size_t n_seqs = cu_seq_lens_cpu.numel() - 1;
  const int32_t* cu_lens = cu_seq_lens_cpu.data_ptr<int32_t>();

  // repeat keys/values if num_heads != num_kv_heads
  torch::Tensor _key = key;
  torch::Tensor _value = value;
  const auto n_heads = query.size(1);
  const auto n_kv_heads = key.size(1);
  if (n_heads != n_kv_heads) {
    GCHECK(n_heads % n_kv_heads == 0);
    const auto num_goups = n_heads / n_kv_heads;
    _key = _key.repeat_interleave(/*repeats=*/num_goups, /*dim=*/1);
    _value = _value.repeat_interleave(/*repeats=*/num_goups, /*dim=*/1);
  }

  for (int64_t i = 0; i < n_seqs; ++i) {
    // calaculate attention for each sequence
    const int32_t start_idx = cu_lens[i];
    const int32_t end_idx = cu_lens[i + 1];
    const int32_t seq_len = end_idx - start_idx;

    // create attention mask based on sequence length
    torch::Tensor mask;
    if (seq_len > 1) {
      mask = torch::full({1, seq_len, seq_len}, negative_infinity);
      mask = torch::triu(mask, /*diagonal=*/1).type_as(query);

      if (alibi_slopes) {
        torch::Tensor slopes = alibi_slopes.value();
        GCHECK(slopes.size(0) == n_heads);

        // calculate alibi attention mask
        auto bias = torch::arange(0, seq_len, query.options());
        bias = bias.unsqueeze(/*dim=*/0) - bias.unsqueeze(/*dim=*/1);
        bias = bias.expand({n_heads, seq_len, seq_len});
        bias = bias * slopes.view({n_heads, 1, 1});

        mask = mask + bias;
      }
    }

    const auto attn = masked_self_attention(
        query.slice(/*dim=*/0, /*start=*/start_idx, /*end=*/end_idx),
        _key.slice(/*dim=*/0, /*start=*/start_idx, /*end=*/end_idx),
        _value.slice(/*dim=*/0, /*start=*/start_idx, /*end=*/end_idx),
        mask,
        scale);
    output.index_put_({Slice(start_idx, end_idx), Slice(), Slice()}, attn);
  }
}

void varlen_masked_self_attention_cuda(
    const torch::Tensor& query,        // [n_tokens, n_heads, head_dim]
    const torch::Tensor& key,          // [n_tokens, n_kv_heads, head_dim]
    const torch::Tensor& value,        // [n_tokens, n_kv_heads, head_dim]
    const torch::Tensor& cu_seq_lens,  // [n_seq + 1]
    torch::optional<torch::Tensor> alibi_slopes,  // [n_heads]
    int32_t max_seq_len,                          // maximum sequence length
    float scale,                                  // scale for softmax
    torch::Tensor& output) {
  auto query_ = query;
  torch::optional<at::Tensor> out = output;
  mha_varlen_fwd_kvcache(query_,
                         key,
                         value,
                         /*knew=*/torch::nullopt,
                         /*vnew=*/torch::nullopt,
                         out,
                         cu_seq_lens,
                         cu_seq_lens,
                         /*cu_seqlens_knew=*/torch::nullopt,
                         /*block_table=*/torch::nullopt,
                         alibi_slopes,
                         max_seq_len,
                         max_seq_len,
                         /*softmax_scale=*/scale,
                         /*is_causal=*/true,
                         /*window_size_left=*/-1,
                         /*window_size_right=*/0);
}

void single_query_masked_self_attention_generic(
    const KVCache& kv_cache,            // where to get key and value
    const torch::Tensor& query,         // [n_tokens/n_seq, n_heads, head_dim]
    const torch::Tensor& block_tables,  // [n_tokens, num_blocks]
    const torch::Tensor& context_lens,  // [n_tokens]
    torch::optional<torch::Tensor> alibi_slopes,  // [n_heads]
    int32_t /*max_context_len*/,                  // maximum context length
    float scale,                                  // scale for softmax
    torch::Tensor& output) {
  const auto n_seq = query.size(0);
  // process each sequence
  // don't need attention mask for single token
  for (int64_t i = 0; i < n_seq; ++i) {
    // [1, n_heads, head_dim]
    const auto q = query[i].unsqueeze(0);
    const auto block_table = block_tables[i];
    const auto context_len = context_lens[i].item<int>();
    // fetch keys/values from cache
    auto [k, v] = kv_cache.get_kv_cache(block_table, context_len);

    // repeat keys/values if num_heads != num_kv_heads
    const auto n_heads = query.size(1);
    const auto n_kv_heads = k.size(1);
    if (n_heads != n_kv_heads) {
      GCHECK(n_heads % n_kv_heads == 0);
      const auto n_goups = n_heads / n_kv_heads;
      k = k.repeat_interleave(/*repeats=*/n_goups, /*dim=*/1);
      v = v.repeat_interleave(/*repeats=*/n_goups, /*dim=*/1);
    }

    torch::Tensor alibi_bias;
    if (alibi_slopes) {
      torch::Tensor slopes = alibi_slopes.value();

      // prepare alibi attention mask
      auto bias = torch::arange(0, context_len, query.options());
      bias -= (context_len - 1);
      // => [n_heads, 1, context_len]
      alibi_bias =
          bias.view({1, 1, context_len}) * slopes.view({n_heads, 1, 1});
    }

    const auto attn = masked_self_attention(q, k, v, alibi_bias, scale);
    output.index_put_({i, Slice(), Slice()}, attn);
  }
}

void single_query_masked_self_attention_cuda(
    const KVCache& kv_cache,  // where to get key and value
    int32_t n_kv_heads,
    const torch::Tensor& query,         // [n_tokens/n_seq, n_heads, head_dim]
    const torch::Tensor& block_tables,  // [n_tokens, num_blocks]
    const torch::Tensor& context_lens,  // [n_tokens]
    torch::optional<torch::Tensor> alibi_slopes,  // [n_heads]
    int32_t max_context_len,                      // maximum context length
    float scale,                                  // scale for softmax
    torch::Tensor& output) {
  auto [key_cache, value_cache] = kv_cache.get_kv_cache();
  static constexpr int32_t kPartitionSize = 512;
  const auto n_seq = query.size(0);
  const auto n_heads = query.size(1);
  const auto head_dim = query.size(2);
  const auto block_size = static_cast<int32_t>(key_cache.size(3));

  // make a 'copy' of variable since the api is using non-const reference
  torch::Tensor _query = query;
  torch::Tensor _block_tables = block_tables;
  torch::Tensor _context_lens = context_lens;

  // Adapted from:
  // https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/attention.py#L253
  // round up partition number
  const auto num_partitions =
      (max_context_len + kPartitionSize - 1) / kPartitionSize;

  // For context len > 8192, use V2 kernel to avoid shared memory shortage.
  const bool use_v1 =
      (max_context_len <= 8192) &&
      (num_partitions == 1 || (n_seq * n_heads) > kPartitionSize);
  // Use the same simple heuristic as vllm to decide whether to use v1.
  if (!FLAGS_force_use_paged_attention_v2 && use_v1) {
    paged_attention_v1(output,
                       _query,
                       key_cache,
                       value_cache,
                       n_kv_heads,
                       scale,
                       _block_tables,
                       _context_lens,
                       block_size,
                       max_context_len,
                       alibi_slopes);
  } else {
    GCHECK(kPartitionSize % block_size == 0);
    auto tmp_out = torch::empty({n_seq, n_heads, num_partitions, head_dim},
                                output.options());
    auto exp_sums =
        torch::empty({n_seq, n_heads, num_partitions},
                     torch::dtype(torch::kFloat32).device(output.device()));
    auto max_logits = torch::empty_like(exp_sums);

    paged_attention_v2(output,
                       exp_sums,
                       max_logits,
                       tmp_out,
                       _query,
                       key_cache,
                       value_cache,
                       n_kv_heads,
                       scale,
                       _block_tables,
                       _context_lens,
                       block_size,
                       max_context_len,
                       alibi_slopes);
  }
}

void masked_self_attention_cuda(
    const torch::Tensor& query,          // [n_q_tokens, n_heads, head_dim]
    const torch::Tensor& key,            // [..., n_kv_heads, head_dim]
    const torch::Tensor& value,          // [..., n_kv_heads, head_dim]
    const torch::Tensor& q_cu_seq_lens,  // [n_seq + 1]
    const torch::Tensor& k_cu_seq_lens,  // [n_seq + 1]
    const torch::optional<torch::Tensor> block_tables,  // [n_seq, max_n_blocks]
    const torch::optional<torch::Tensor> alibi_slopes,  // [n_heads]
    int32_t q_max_seq_len,  // maximum sequence length for Q
    int32_t k_max_seq_len,  // maximum sequence length for K/V
    float scale,
    torch::Tensor& output) {
  auto query_ = query;
  torch::optional<at::Tensor> out = output;
  mha_varlen_fwd_kvcache(query_,
                         key,
                         value,
                         /*knew=*/torch::nullopt,
                         /*vnew=*/torch::nullopt,
                         out,
                         q_cu_seq_lens,
                         k_cu_seq_lens,
                         /*cu_seqlens_knew=*/torch::nullopt,
                         block_tables,
                         alibi_slopes,
                         q_max_seq_len,
                         k_max_seq_len,
                         /*softmax_scale=*/scale,
                         /*is_causal=*/true,
                         /*window_size_left=*/-1,
                         /*window_size_right=*/0);
}

void masked_self_attention_with_newk_cuda(
    const torch::Tensor& query,  // [n_q_tokens, n_heads, head_dim]
    const torch::Tensor& kcache,
    const torch::Tensor& vcache,
    torch::optional<torch::Tensor> knew,  // [new_tokens, n_kv_heads, head_dim]
    torch::optional<torch::Tensor> vnew,  // [new_tokens, n_kv_heads, head_dim]
    const torch::Tensor& q_cu_seq_lens,   // [n_seq + 1]
    const torch::Tensor& k_cu_seq_lens,   // [n_seq + 1]
    torch::optional<torch::Tensor> knew_cu_seq_lens,  // [n_seq + 1]
    torch::optional<torch::Tensor> block_tables,      // [n_seq, max_n_blocks]
    torch::optional<torch::Tensor> alibi_slopes,      // [n_heads]
    int32_t q_max_seq_len,  // maximum sequence length for Q
    int32_t k_max_seq_len,  // maximum sequence length for K/V
    float scale,
    torch::Tensor& output) {
  auto query_ = query;
  torch::optional<at::Tensor> out = output;
  mha_varlen_fwd_kvcache(query_,
                         kcache,
                         vcache,
                         knew,
                         vnew,
                         out,
                         q_cu_seq_lens,
                         k_cu_seq_lens,
                         knew_cu_seq_lens,
                         block_tables,
                         alibi_slopes,
                         q_max_seq_len,
                         k_max_seq_len,
                         /*softmax_scale=*/scale,
                         /*is_causal=*/true,
                         /*window_size_left=*/-1,
                         /*window_size_right=*/0);
}

}  // namespace detail

}  // namespace llm
