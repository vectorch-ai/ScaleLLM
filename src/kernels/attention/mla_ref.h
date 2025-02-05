#pragma once

#include <torch/torch.h>

namespace llm {
// Multi-head latten attention implementation using pytorch
inline torch::Tensor mla_batch_ref(
    torch::Tensor query,  // [batch_size, q_len, n_heads, head_dim]
    torch::Tensor key,    // [batch_size, kv_len, n_kv_heads, head_dim]
    torch::Tensor value,  // [batch_size, kv_len, n_kv_heads, head_dim]
    torch::optional<torch::Tensor> alibi_slopes,  //[n_heads]
    float logits_soft_cap,
    int32_t sliding_window) {
  // const auto q_len = query.size(-3);
  // const auto kv_len = key.size(-3);
  // const auto n_heads = query.size(-2);
  // const auto n_kv_heads = key.size(-2);
  // const auto head_dim = query.size(-1);
  // assert(kv_len >= q_len);

  // if (n_heads != n_kv_heads) {
  //   assert(n_heads % n_kv_heads == 0);
  //   const auto group_size = n_heads / n_kv_heads;
  //   key = key.repeat_interleave(/*repeats=*/group_size, /*dim=*/-2);
  //   value = value.repeat_interleave(/*repeats=*/group_size, /*dim=*/-2);
  // }

  // const float sm_scale = 1.0 / sqrt(head_dim);
  // // query * key => [n_heads, q_seq_len, seq_len]
  // auto scores = torch::einsum("bqhd,bkhd->bhqk",
  //                             {query.to(torch::kFloat), key.to(torch::kFloat)});
  // // apply scale
  // scores *= sm_scale;

  // // apply softcap if needed
  // if (logits_soft_cap != 0.0) {
  //   scores = torch::tanh(scores / logits_soft_cap) * logits_soft_cap;
  // }

  // // apply alibi bias
  // if (alibi_slopes) {
  //   const auto& slopes = alibi_slopes.value();
  //   // calculate alibi attention bias
  //   // since it's causal mask, we can just use [0, 1, ...,, kv_len)
  //   auto distance = torch::arange(0, kv_len, query.options());
  //   // [n_heads, 1, kv_len]
  //   auto bias = distance.view({1, 1, kv_len}) * slopes.view({n_heads, 1, 1});
  //   scores += bias;
  // }

  // auto mask = torch::ones({q_len, kv_len}, torch::kBool);
  // if (sliding_window >= 0) {
  //   // sliding window mask
  //   // returns the upper triangular part of a matrix
  //   mask = torch::triu(mask, /*diagonal=*/kv_len - q_len - sliding_window);
  // }

  // // apply causal mask
  // // causal mask: returns the lower triangular part of a matrix
  // mask = torch::tril(mask, /*diagonal=*/kv_len - q_len).to(query);
  // scores = scores.masked_fill(mask == 0, -INFINITY);

  // // safe softmax
  // scores = torch::softmax(scores, /*dim=*/-1);

  // // score * value => [batch_size, q_len, n_heads, head_dim]
  // return torch::einsum("bhqk,bkhd->bqhd", {scores, value.to(torch::kFloat)})
  //     .type_as(query);
}

}  // namespace llm