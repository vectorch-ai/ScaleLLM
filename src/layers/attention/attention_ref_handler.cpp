#include "attention_ref_handler.h"

#include <gflags/gflags.h>
#include <torch/torch.h>

#include "memory/kv_cache.h"
#include "models/input_parameters.h"

namespace llm {
using torch::indexing::Slice;

namespace {
constexpr float negative_infinity = -std::numeric_limits<float>::infinity();

torch::Tensor masked_self_attention(
    const torch::Tensor& query,  // [q_seq_len, n_heads, head_dim]
    const torch::Tensor& key,    // [k_seq_len, n_heads, head_dim]
    const torch::Tensor& value,  // [k_seq_len, n_heads, head_dim]
    const torch::Tensor& mask,   // [n_heads, q_seq_len, k_seq_len]
    float scale) {
  // => [n_heads, q_seq_len, k_seq_len]
  auto scores = torch::einsum("qhd,khd->hqk",
                              {query.to(torch::kFloat), key.to(torch::kFloat)});
  scores *= scale;
  if (mask.defined()) {
    scores += mask;
  }
  scores = torch::softmax(scores, /*dim=*/-1);
  // => [q_seq_len, n_heads, head_dim]
  return torch::einsum("hqk,khd->qhd", {scores, value.to(torch::kFloat)})
      .type_as(query);
}

// slow reference implementation for varlen_masked_self_attention
void varlen_masked_self_attention(
    const torch::Tensor& query,           // [n_tokens, n_heads, head_dim]
    const torch::Tensor& key,             // [n_tokens, n_kv_heads, head_dim]
    const torch::Tensor& value,           // [n_tokens, n_kv_heads, head_dim]
    const torch::Tensor& q_cu_seq_lens,   // [n_seqs + 1]
    const torch::Tensor& kv_cu_seq_lens,  // [n_seqs + 1]
    const KVCache& kv_cache,              // where to get key and value
    const torch::Tensor& block_tables,    // [n_seqs, max_n_blocks]
    const torch::optional<torch::Tensor> alibi_slopes,  // [n_heads]
    float scale,
    torch::Tensor& output) {
  // same length for key and value
  DCHECK(key.size(0) == value.size(0));

  const auto head_dim = query.size(-1);
  const auto n_heads = query.size(-2);
  const auto n_kv_heads = key.size(-2);

  torch::Tensor q_cu_seq_lens_cpu = q_cu_seq_lens.cpu();
  torch::Tensor kv_cu_seq_lens_cpu = kv_cu_seq_lens.cpu();
  const size_t n_seqs = q_cu_seq_lens_cpu.numel() - 1;
  const int32_t* q_cu_lens = q_cu_seq_lens_cpu.data_ptr<int32_t>();
  const int32_t* kv_cu_lens = kv_cu_seq_lens_cpu.data_ptr<int32_t>();

  // process sequence one by one
  for (int64_t i = 0; i < n_seqs; ++i) {
    // calaculate attention for each sequence
    const int32_t q_start = q_cu_lens[i];
    const int32_t q_end = q_cu_lens[i + 1];
    const int32_t q_len = q_end - q_start;
    const int32_t kv_start = kv_cu_lens[i];
    const int32_t kv_end = kv_cu_lens[i + 1];
    const int32_t kv_len = kv_end - kv_start;

    torch::Tensor _query =
        query.slice(/*dim=*/0, /*start=*/q_start, /*end=*/q_end);
    torch::Tensor _key =
        key.slice(/*dim=*/0, /*start=*/kv_start, /*end=*/kv_end);
    torch::Tensor _value =
        value.slice(/*dim=*/0, /*start=*/kv_start, /*end=*/kv_end);

    CHECK(kv_len >= q_len);
    if (_key.size(0) < kv_len) {
      // featch key and value from kv_cache
      auto [k, v] = kv_cache.get_kv_cache(block_tables[i], kv_len);
      _key = k;
      _value = v;
    }

    // repeat key and value if n_heads > n_kv_heads
    if (n_heads != n_kv_heads) {
      CHECK(n_heads % n_kv_heads == 0);
      const auto num_goups = n_heads / n_kv_heads;
      _key = _key.repeat_interleave(/*repeats=*/num_goups, /*dim=*/-2);
      _value = _value.repeat_interleave(/*repeats=*/num_goups, /*dim=*/-2);
    }

    // causal mask
    // [1, q_len, kv_len]
    torch::Tensor mask = torch::full({1, q_len, kv_len}, negative_infinity);
    mask = torch::triu(mask, /*diagonal=*/kv_len - q_len + 1).type_as(query);

    if (alibi_slopes) {
      torch::Tensor slopes = alibi_slopes.value();
      CHECK(slopes.size(0) == n_heads);

      // calculate alibi attention mask
      auto bias = torch::arange(0, kv_len, query.options());
      // [kv_len, kv_len]
      bias = bias.unsqueeze(/*dim=*/0) - bias.unsqueeze(/*dim=*/1);
      // [q_len, kv_len]
      bias = bias.slice(/*dim=*/0, /*start=*/kv_len - q_len, /*end=*/kv_len);
      // [n_heads, q_len, kv_len]
      bias = bias.expand({n_heads, q_len, kv_len});
      bias = bias * slopes.view({n_heads, 1, 1});
      mask = mask + bias;
    }

    const auto attn = masked_self_attention(_query, _key, _value, mask, scale);
    output.index_put_({Slice(q_start, q_end), Slice(), Slice()}, attn);
  }
}

}  // namespace

AttentionRefHandler::AttentionRefHandler(
    float scale,
    torch::optional<torch::Tensor> alibi_slopes)
    : scale_(scale), alibi_slopes_(alibi_slopes) {}

// batch prefill for attention, optimized for prefill stage
void AttentionRefHandler::batch_prefill(
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
  batch_prefill(query,
                key,
                value,
                input_params.q_cu_seq_lens,
                input_params.kv_cu_seq_lens,
                output);
}

void AttentionRefHandler::batch_prefill(
    const torch::Tensor& query,           // [n_tokens, n_heads, head_dim]
    const torch::Tensor& key,             // [n_tokens, n_kv_heads, head_dim]
    const torch::Tensor& value,           // [n_tokens, n_kv_heads, head_dim]
    const torch::Tensor& q_cu_seq_lens,   // [n_seqs + 1]
    const torch::Tensor& kv_cu_seq_lens,  // [n_seqs + 1]
    torch::Tensor& output) {
  varlen_masked_self_attention(query,
                               key,
                               value,
                               q_cu_seq_lens,
                               kv_cu_seq_lens,
                               /*kv_cache=*/{},
                               /*block_tables=*/{},
                               alibi_slopes_,
                               scale_,
                               output);
}

// batch decode for attention, optimized for decode stage
// support multiple queries: one sequence with multiple query tokens
void AttentionRefHandler::batch_decode(
    const torch::Tensor& query,  // [n_tokens, n_heads, head_dim]
    const torch::Tensor& key,    // [n_tokens, n_kv_heads, head_dim]
    const torch::Tensor& value,  // [n_tokens, n_kv_heads, head_dim]
    KVCache& kv_cache,           // where to store and retrieval key and value
    const InputParameters& input_params,  // input paras used for attention
    torch::Tensor& output) {
  // append key and value to kv_cache
  kv_cache.set_kv_cache(input_params.new_cache_slots, key, value);

  varlen_masked_self_attention(query,
                               key,
                               value,
                               input_params.q_cu_seq_lens,
                               input_params.kv_cu_seq_lens,
                               kv_cache,
                               input_params.block_tables,
                               alibi_slopes_,
                               scale_,
                               output);
}

}  // namespace llm
