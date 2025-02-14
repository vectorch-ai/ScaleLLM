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
  auto scores = torch::einsum("bqhr,bkr->bqhk", {q_, kv_}) +
                torch::einsum("bqhp,bkp->bqhk", {q_rope_, k_rope_});
  // apply scale
  scores *= sm_scale;

  // safe softmax
  scores = torch::softmax(scores, /*dim=*/-1);

  // score * value => [batch_size, q_len, n_heads, kv_lora_rank]
  return torch::einsum("bqhk,bkr->bqhr", {scores, kv_}).type_as(q);
}

}  // namespace llm