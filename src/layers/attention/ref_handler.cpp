#include "ref_handler.h"

#include <torch/torch.h>

#include "memory/kv_cache.h"
#include "models/parameters.h"

namespace llm {
using ISlice = torch::indexing::Slice;

namespace {
torch::Tensor masked_self_attention(
    const torch::Tensor& query,         // [q_seq_len, n_heads, head_dim]
    const torch::Tensor& key,           // [k_seq_len, n_heads, head_dim]
    const torch::Tensor& value,         // [k_seq_len, n_heads, head_dim]
    const torch::Tensor& alibi_biases,  // [n_heads, q_seq_len, k_seq_len]
    const torch::Tensor& mask,          // [n_heads, q_seq_len, k_seq_len]
    float sm_scale) {
  // => [n_heads, q_seq_len, k_seq_len]
  auto scores = torch::einsum("qhd,khd->hqk",
                              {query.to(torch::kFloat), key.to(torch::kFloat)});
  // apply scale
  scores *= sm_scale;

  // scores /= softcap;
  // scores = scores.tanh()
  // scores *= softcap;

  // add alibi biases to attention scores
  if (alibi_biases.defined()) {
    scores += alibi_biases;
  }
  // apply causal mask
  if (mask.defined()) {
    scores = scores.masked_fill(mask == 0, -INFINITY);
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

    // repeat key and value if n_heads > n_kv_heads
    if (n_heads != n_kv_heads) {
      CHECK(n_heads % n_kv_heads == 0);
      const auto num_goups = n_heads / n_kv_heads;
      _key = _key.repeat_interleave(/*repeats=*/num_goups, /*dim=*/-2);
      _value = _value.repeat_interleave(/*repeats=*/num_goups, /*dim=*/-2);
    }

    // causal mask
    // [1, q_len, kv_len]
    torch::Tensor mask = torch::ones({1, q_len, kv_len}, torch::kBool);
    // returns the lower triangular part of a matrix
    mask = torch::tril(mask, /*diagonal=*/kv_len - q_len).to(query);

    torch::Tensor bias;
    if (alibi_slopes) {
      const torch::Tensor& slopes = alibi_slopes.value();
      CHECK(slopes.size(0) == n_heads);

      // calculate alibi attention bias
      // since it's causal mask, we can just use [0, 1, ...,, kv_len)
      auto distance = torch::arange(0, kv_len, query.options());
      // [n_heads, 1, kv_len]
      bias = distance.view({1, 1, kv_len}) * slopes.view({n_heads, 1, 1});
    }

    const auto attn =
        masked_self_attention(_query, _key, _value, bias, mask, scale);
    output.index_put_({ISlice(q_start, q_end), ISlice(), ISlice()}, attn);
  }
}

}  // namespace

RefHandler::RefHandler(float scale,
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

RefHandler::RefHandler(float scale, torch::optional<torch::Tensor> alibi_slopes)
    : sm_scale_(scale), alibi_slopes_(alibi_slopes) {}

std::tuple<torch::Tensor, torch::Tensor> RefHandler::apply_pos_emb(
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
void RefHandler::batch_prefill(
    const torch::Tensor& query,           // [n_tokens, n_heads, head_dim]
    const torch::Tensor& key,             // [n_tokens, n_kv_heads, head_dim]
    const torch::Tensor& value,           // [n_tokens, n_kv_heads, head_dim]
    const InputParameters& input_params,  // input paras used for attention
    int32_t sliding_window,               // sliding window size
    torch::Tensor& output) {
  // TODO: add sliding window support
  // don't use kv cache in prefill stage
  varlen_masked_self_attention(query,
                               key,
                               value,
                               input_params.q_cu_seq_lens,
                               input_params.kv_cu_seq_lens,
                               alibi_slopes_,
                               sm_scale_,
                               output);
}

// batch decode for attention, optimized for decode stage
// support multiple queries: one sequence with multiple query tokens
void RefHandler::batch_decode(
    const torch::Tensor& query,           // [n_tokens, n_heads, head_dim]
    const KVCache& kv_cache,              // where to retrieval key and value
    const InputParameters& input_params,  // input paras used for attention
    int32_t sliding_window,               // sliding window size
    torch::Tensor& output) {
  // retrieval key and value from kv_cache
  auto [key, value] = kv_cache.get_kv_cache(input_params.block_tables,
                                            input_params.kv_cu_seq_lens);
  // TODO: add sliding window support
  varlen_masked_self_attention(query,
                               key,
                               value,
                               input_params.q_cu_seq_lens,
                               input_params.kv_cu_seq_lens,
                               alibi_slopes_,
                               sm_scale_,
                               output);
}

// append key and value to kv_cache
void RefHandler::append_kv_cache(
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
