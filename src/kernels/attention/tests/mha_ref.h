#pragma once

#include <torch/torch.h>

namespace llm {
// Multi-head attention implementation using pytorch
inline torch::Tensor mha_batch_ref(
    torch::Tensor query,  // [batch_size, q_len, n_heads, head_dim]
    torch::Tensor key,    // [batch_size, kv_len, n_kv_heads, head_dim]
    torch::Tensor value,  // [batch_size, kv_len, n_kv_heads, head_dim]
    torch::optional<torch::Tensor> alibi_slopes,  //[n_heads]
    float logits_soft_cap,
    int32_t sliding_window) {
  const auto q_len = query.size(-3);
  const auto kv_len = key.size(-3);
  const auto n_heads = query.size(-2);
  const auto n_kv_heads = key.size(-2);
  const auto head_dim = query.size(-1);
  assert(kv_len >= q_len);

  if (n_heads != n_kv_heads) {
    assert(n_heads % n_kv_heads == 0);
    const auto group_size = n_heads / n_kv_heads;
    key = key.repeat_interleave(/*repeats=*/group_size, /*dim=*/-2);
    value = value.repeat_interleave(/*repeats=*/group_size, /*dim=*/-2);
  }

  const float sm_scale = 1.0 / sqrt(head_dim);
  // query * key => [n_heads, q_seq_len, seq_len]
  auto scores = torch::einsum("bqhd,bkhd->bhqk",
                              {query.to(torch::kFloat), key.to(torch::kFloat)});
  // apply scale
  scores *= sm_scale;

  // apply softcap if needed
  if (logits_soft_cap != 0.0) {
    scores = torch::tanh(scores / logits_soft_cap) * logits_soft_cap;
  }

  // apply alibi bias
  if (alibi_slopes) {
    const auto& slopes = alibi_slopes.value();
    // calculate alibi attention bias
    // since it's causal mask, we can just use [0, 1, ...,, kv_len)
    auto distance = torch::arange(0, kv_len, query.options());
    // [n_heads, 1, kv_len]
    auto bias = distance.view({1, 1, kv_len}) * slopes.view({n_heads, 1, 1});
    scores += bias;
  }

  auto mask = torch::ones({q_len, kv_len}, torch::kBool);
  if (sliding_window >= 0) {
    // sliding window mask
    // returns the upper triangular part of a matrix
    mask = torch::triu(mask, /*diagonal=*/kv_len - q_len - sliding_window);
  }

  // apply causal mask
  // causal mask: returns the lower triangular part of a matrix
  mask = torch::tril(mask, /*diagonal=*/kv_len - q_len).to(query);
  scores = scores.masked_fill(mask == 0, -INFINITY);

  // safe softmax
  scores = torch::softmax(scores, /*dim=*/-1);

  // score * value => [batch_size, q_len, n_heads, head_dim]
  return torch::einsum("bhqk,bkhd->bqhd", {scores, value.to(torch::kFloat)})
      .type_as(query);
}

inline torch::Tensor mha_ref(
    torch::Tensor query,  // [q_len, n_heads, head_dim]
    torch::Tensor key,    // [kv_len, n_kv_heads, head_dim]
    torch::Tensor value,  // [kv_len, n_kv_heads, head_dim]
    torch::optional<torch::Tensor> alibi_slopes,  //[n_heads]
    float logits_soft_cap,
    int32_t sliding_window) {
  const auto q_len = query.size(-3);
  const auto kv_len = key.size(-3);
  const auto n_heads = query.size(-2);
  const auto n_kv_heads = key.size(-2);
  const auto head_dim = query.size(-1);
  assert(kv_len >= q_len);

  if (n_heads != n_kv_heads) {
    assert(n_heads % n_kv_heads == 0);
    const auto group_size = n_heads / n_kv_heads;
    key = key.repeat_interleave(/*repeats=*/group_size, /*dim=*/-2);
    value = value.repeat_interleave(/*repeats=*/group_size, /*dim=*/-2);
  }

  const float sm_scale = 1.0 / sqrt(head_dim);
  // query * key => [n_heads, q_len, kv_len]
  auto scores = torch::einsum("qhd,khd->hqk",
                              {query.to(torch::kFloat), key.to(torch::kFloat)});
  // apply scale
  scores *= sm_scale;

  // apply softcap if needed
  if (logits_soft_cap != 0.0) {
    scores = torch::tanh(scores / logits_soft_cap) * logits_soft_cap;
  }

  // apply alibi bias
  if (alibi_slopes) {
    const auto& slopes = alibi_slopes.value();
    // calculate alibi attention bias
    // since it's causal mask, we can just use [0, 1, ...,, kv_len)
    auto distance = torch::arange(0, kv_len, query.options());
    // [n_heads, 1, kv_len]
    auto bias = distance.view({1, 1, kv_len}) * slopes.view({n_heads, 1, 1});
    scores += bias;
  }

  auto mask = torch::ones({q_len, kv_len}, torch::kBool);
  if (sliding_window >= 0) {
    // sliding window mask
    // returns the upper triangular part of a matrix
    mask = torch::triu(mask, /*diagonal=*/kv_len - q_len - sliding_window);
  }

  // apply causal mask
  // causal mask: returns the lower triangular part of a matrix
  mask = torch::tril(mask, /*diagonal=*/kv_len - q_len).to(query);
  scores = scores.masked_fill(mask == 0, -INFINITY);

  // safe softmax
  scores = torch::softmax(scores, /*dim=*/-1);

  // score * value => [q_len, n_heads, head_dim]
  return torch::einsum("hqk,khd->qhd", {scores, value.to(torch::kFloat)})
      .type_as(query);
}

inline torch::Tensor mha_varlen_ref(
    torch::Tensor query,       // [q_len, n_heads, head_dim]
    torch::Tensor key,         // [kv_len, n_kv_heads, head_dim]
    torch::Tensor value,       // [kv_len, n_kv_heads, head_dim]
    torch::Tensor q_cu_lens,   // [batch_size + 1]
    torch::Tensor kv_cu_lens,  // [batch_size + 1]
    torch::optional<torch::Tensor> alibi_slopes,  //[n_heads]
    float logits_soft_cap,
    int32_t sliding_window) {
  torch::Tensor q_cu_lens_cpu = q_cu_lens.cpu();
  torch::Tensor kv_cu_seq_lens_cpu = kv_cu_lens.cpu();
  const size_t n_seqs = q_cu_lens_cpu.numel() - 1;
  const int32_t* q_cu_lens_ptr = q_cu_lens_cpu.data_ptr<int32_t>();
  const int32_t* kv_cu_lens_ptr = kv_cu_seq_lens_cpu.data_ptr<int32_t>();

  std::vector<torch::Tensor> out_list;
  // process sequence one by one
  for (int64_t i = 0; i < n_seqs; ++i) {
    // calaculate attention for each sequence
    const int32_t q_start = q_cu_lens_ptr[i];
    const int32_t q_end = q_cu_lens_ptr[i + 1];
    const int32_t kv_start = kv_cu_lens_ptr[i];
    const int32_t kv_end = kv_cu_lens_ptr[i + 1];

    torch::Tensor q = query.slice(/*dim=*/0, /*start=*/q_start, /*end=*/q_end);
    torch::Tensor k = key.slice(/*dim=*/0, /*start=*/kv_start, /*end=*/kv_end);
    torch::Tensor v =
        value.slice(/*dim=*/0, /*start=*/kv_start, /*end=*/kv_end);

    auto output =
        mha_ref(q, k, v, alibi_slopes, logits_soft_cap, sliding_window);
    out_list.push_back(output);
  }
  return torch::cat(out_list, /*dim=*/0);
}
}  // namespace llm
