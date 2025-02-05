#pragma once

#include <torch/torch.h>

namespace llm {
// Multi-head latten attention implementation using pytorch
// reference implementation:
// https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py#L477
inline torch::Tensor mla_batch_ref(
    torch::Tensor q,       // [batch, q_len, n_heads, kv_lora_rank]
    torch::Tensor q_rope,  // [batch, q_len, n_heads, qk_rope_head_dim]
    torch::Tensor kv,      // [batch, kv_len, kv_lora_rank]
    torch::Tensor k_rope,  // [batch, kv_len, qk_rope_head_dim]
    float sm_scale) {
  const auto q_len = q.size(-3);
  const auto n_heads = q.size(-2);
  const auto kv_len = kv.size(-2);
  const auto kv_lora_rank = kv.size(-1);
  const auto qk_rope_head_dim = q_rope.size(-1);
  assert(kv_len >= q_len);

  // query * key => [batch, q_len, n_heads, kv_len]
  auto scores = torch::einsum("bqhr,bkr->bqhk", {q, kv}) +
                torch::einsum("bqhp,bkp->bqhk", {q_rope, k_rope});
  // apply scale
  scores *= sm_scale;

  // safe softmax
  scores = scores.softmax(/*dim=*/-1, /*dtype=*/torch::kFloat).type_as(q);

  // score * value => [batch_size, q_len, n_heads, kv_lora_rank]
  return torch::einsum("bqhk,bkr->bqhr", {scores, kv});
}

}  // namespace llm