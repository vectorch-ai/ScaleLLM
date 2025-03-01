#pragma once

#include <torch/torch.h>

namespace llm {
// Multi-head latent attention implementation using pytorch
// reference implementation:
// https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py#L477
inline torch::Tensor mla_batch_ref(
    torch::Tensor q,       // [batch, q_len, n_heads, head_dim]
    torch::Tensor kv,      // [batch, kv_len, head_dim]
    torch::Tensor q_rope,  // [batch, q_len, n_heads, rope_head_dim]
    torch::Tensor k_rope,  // [batch, kv_len, rope_head_dim]
    float sm_scale) {
  const auto q_len = q.size(-3);
  const auto n_heads = q.size(-2);
  const auto kv_len = kv.size(-2);
  const auto kv_lora_rank = kv.size(-1);
  const auto qk_rope_head_dim = q_rope.size(-1);
  assert(kv_len >= q_len);

  // use float32 for better precision
  auto q_ = q.to(torch::kFloat);
  auto kv_ = kv.to(torch::kFloat);
  auto q_rope_ = q_rope.to(torch::kFloat);
  auto k_rope_ = k_rope.to(torch::kFloat);

  // query * key => [batch, q_len, n_heads, kv_len]
  auto scores = torch::einsum("bqhr,bkr->bhqk", {q_, kv_}) +
                torch::einsum("bqhp,bkp->bhqk", {q_rope_, k_rope_});
  // apply scale
  scores *= sm_scale;

  // apply causal mask
  auto mask = torch::ones({q_len, kv_len}, torch::kBool);
  // causal mask: returns the lower triangular part of a matrix
  mask = torch::tril(mask, /*diagonal=*/kv_len - q_len).to(q);
  scores = scores.masked_fill(mask == 0, -INFINITY);

  // safe softmax
  scores = torch::softmax(scores, /*dim=*/-1);

  // score * value => [batch_size, q_len, n_heads, head_dim]
  return torch::einsum("bhqk,bkr->bqhr", {scores, kv_}).type_as(q);
}

inline torch::Tensor mla_ref(
    torch::Tensor q,       // [q_len, n_heads, head_dim]
    torch::Tensor kv,      // [kv_len, head_dim]
    torch::Tensor q_rope,  // [q_len, n_heads, rope_head_dim]
    torch::Tensor k_rope,  // [kv_len, rope_head_dim]
    float sm_scale) {
  const auto q_len = q.size(-3);
  const auto n_heads = q.size(-2);
  const auto kv_len = kv.size(-2);
  const auto kv_lora_rank = kv.size(-1);
  const auto qk_rope_head_dim = q_rope.size(-1);
  assert(kv_len >= q_len);

  // use float32 for better precision
  auto q_ = q.to(torch::kFloat);
  auto kv_ = kv.to(torch::kFloat);
  auto q_rope_ = q_rope.to(torch::kFloat);
  auto k_rope_ = k_rope.to(torch::kFloat);

  // query * key => [q_len, n_heads, kv_len]
  auto scores = torch::einsum("qhr,kr->hqk", {q_, kv_}) +
                torch::einsum("qhp,kp->hqk", {q_rope_, k_rope_});
  // apply scale
  scores *= sm_scale;

  // apply causal mask
  auto mask = torch::ones({q_len, kv_len}, torch::kBool);
  // causal mask: returns the lower triangular part of a matrix
  mask = torch::tril(mask, /*diagonal=*/kv_len - q_len).to(q);
  scores = scores.masked_fill(mask == 0, -INFINITY);

  // safe softmax
  scores = torch::softmax(scores, /*dim=*/-1);

  // score * value => [q_len, n_heads, head_dim]
  return torch::einsum("hqk,kr->qhr", {scores, kv_}).type_as(q);
}

inline torch::Tensor mla_varlen_ref(
    torch::Tensor q,           // [q_len, n_heads, head_dim]
    torch::Tensor kv,          // [kv_len, head_dim]
    torch::Tensor q_rope,      // [q_len, n_heads, rope_head_dim]
    torch::Tensor k_rope,      // [kv_len, rope_head_dim]
    torch::Tensor q_cu_lens,   // [batch_size + 1]
    torch::Tensor kv_cu_lens,  // [batch_size + 1]
    float sm_scale) {
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

    torch::Tensor q_ = q.slice(/*dim=*/0, /*start=*/q_start, /*end=*/q_end);
    torch::Tensor kv_ = kv.slice(/*dim=*/0, /*start=*/kv_start, /*end=*/kv_end);

    torch::Tensor q_rope_ =
        q_rope.slice(/*dim=*/0, /*start=*/q_start, /*end=*/q_end);
    torch::Tensor k_rope_ =
        k_rope.slice(/*dim=*/0, /*start=*/kv_start, /*end=*/kv_end);

    auto output = mla_ref(q_, kv_, q_rope_, k_rope_, sm_scale);
    out_list.push_back(output);
  }
  return torch::cat(out_list, /*dim=*/0);
}

}  // namespace llm