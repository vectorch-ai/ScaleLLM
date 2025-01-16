#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>

#include "attention_launch_sm80.cuh"
#include "attention_params.h"
#include "cute/layout.hpp"
#include "static_dispatch.h"

namespace llm {
namespace {
// Multi-head attention implementation using pytorch
torch::Tensor attention_ref(
    torch::Tensor query,  // [batch_size, n_heads, q_len, head_dim]
    torch::Tensor key,    // [batch_size, n_kv_heads, kv_len, head_dim]
    torch::Tensor value,  // [batch_size, n_kv_heads, kv_len, head_dim]
    torch::optional<torch::Tensor> alibi_slopes,  //[n_heads]
    float logits_soft_cap,
    int32_t sliding_window) {
  const auto q_len = query.size(2);
  const auto kv_len = key.size(2);
  const auto n_heads = query.size(1);
  const auto n_kv_heads = key.size(1);
  const auto head_dim = query.size(3);
  assert(kv_len >= q_len);

  if (n_heads != n_kv_heads) {
    assert(n_heads % n_kv_heads == 0);
    const auto group_size = n_heads / n_kv_heads;
    key = key.repeat_interleave(/*repeats=*/group_size, /*dim=*/-3);
    value = value.repeat_interleave(/*repeats=*/group_size, /*dim=*/-3);
  }

  const float sm_scale = 1.0 / sqrt(head_dim);
  // query * key => [n_heads, q_seq_len, seq_len]
  auto scores = torch::einsum("bhqd,bhkd->bhqk",
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

  // score * value => [batch_size, n_heads, q_seq_len, head_dim]
  return torch::einsum("bhqk,bhkd->bhqd", {scores, value.to(torch::kFloat)})
      .type_as(query);
}

torch::Tensor attention_sm80(
    torch::Tensor query,  // [batch_size, n_heads, q_len, head_dim]
    torch::Tensor key,    // [batch_size, n_kv_heads, kv_len, head_dim]
    torch::Tensor value,  // [batch_size, n_kv_heads, kv_len, head_dim]
    torch::optional<torch::Tensor> alibi_slopes,  //[n_heads]
    float logits_soft_cap,
    int32_t sliding_window,
    int32_t max_q_len) {
  const auto batch_size = query.size(0);
  const auto n_heads = query.size(1);
  const auto n_kv_heads = key.size(1);
  const auto q_len = query.size(2);
  const auto kv_len = key.size(2);
  const auto head_dim = query.size(3);

  const auto h_stride = query.stride(1);
  const auto kv_h_stride = key.stride(1);

  auto out = torch::empty_like(query);

  const float sm_scale = 1.0 / sqrt(head_dim);

  // construct attention params
  AttentionParams params;
  params.q_ptr = query.const_data_ptr();
  params.q_stride =
      make_stride(query.stride(0), query.stride(1), query.stride(2));
  params.k_ptr = key.const_data_ptr();
  params.k_stride = make_stride(key.stride(0), key.stride(1), key.stride(2));
  params.v_ptr = value.const_data_ptr();
  params.v_stride =
      make_stride(value.stride(0), value.stride(1), value.stride(2));
  params.o_ptr = out.mutable_data_ptr();
  params.o_stride = make_stride(out.stride(0), out.stride(1), out.stride(2));
  params.alibi_slopes_ptr = alibi_slopes.has_value()
                                ? alibi_slopes.value().const_data_ptr<float>()
                                : nullptr;

  params.batch_size = batch_size;
  params.max_q_len = max_q_len;
  params.n_heads = n_heads;
  params.n_kv_heads = n_kv_heads;
  params.q_len = q_len;
  params.kv_len = kv_len;
  params.head_dim = head_dim;
  params.sm_scale = sm_scale;
  params.logits_soft_cap = logits_soft_cap;
  params.sliding_window = sliding_window;

  DISPATCH_TORCH_DTYPE(query.dtype(), QTYPE, [&] {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, [&] {
      run_attention_kernel_sm80<QTYPE, HEAD_DIM>(params);
    });
  });
  return out;
}

}  // namespace

class AttentionKernelTest
    : public ::testing::TestWithParam<std::tuple<int64_t /*batch_size*/,
                                                 int64_t /*q_len*/,
                                                 int64_t /*kv_len*/,
                                                 int64_t /*n_heads*/,
                                                 int64_t /*n_kv_heads*/,
                                                 int64_t /*head_dim*/,
                                                 float /*logits_soft_cap*/,
                                                 bool /*alibi*/,
                                                 int32_t /*sliding_window*/>> {
 public:
  void SetUp() override {
    // Set random seed for test stability
    torch::manual_seed(0);
  }
};

TEST_P(AttentionKernelTest, MHA) {
  const auto [batch_size,
              q_len,
              kv_len,
              n_heads,
              n_kv_heads,
              head_dim,
              logits_soft_cap,
              alibi,
              sliding_window] = GetParam();

  const auto options = torch::dtype(torch::kHalf).device(torch::kCUDA);

  // construct non-contiguous query, key and value
  const auto data = torch::randn(
      {batch_size, n_heads + 2 * n_kv_heads, q_len, head_dim}, options);
  const auto qkv =
      data.split(/*split_size=*/{n_heads, n_kv_heads, n_kv_heads}, /*dim=*/1);
  const auto& query = qkv[0];
  const auto& key = qkv[1];
  const auto& value = qkv[2];

  torch::optional<torch::Tensor> alibi_slopes;
  if (alibi) {
    alibi_slopes = torch::rand(
        {n_heads}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  }

  auto ref_out = attention_ref(
      query, key, value, alibi_slopes, logits_soft_cap, sliding_window);
  auto out = attention_sm80(
      query, key, value, alibi_slopes, logits_soft_cap, sliding_window, q_len);

  EXPECT_TRUE(torch::allclose(out, ref_out, /*rtol=*/1e-3, /*atol=*/1e-3));
}

INSTANTIATE_TEST_SUITE_P(
    MHA,
    AttentionKernelTest,
    ::testing::Combine(
        ::testing::Values(1, 2, 4),                          // batch_size
        ::testing::Values(1, 62, 125),                       // q_len
        ::testing::Values(127, 287, 1000),                   // kv_len
        ::testing::Values(6),                                // n_heads
        ::testing::Values(6 /*mha*/, 3 /*gqa*/, 1 /*mqa*/),  // n_kv_heads
        ::testing::Values(64),                               // head_dim
        ::testing::Values(0.0, 50.0),                        // logits_soft_cap
        ::testing::Values(false, true),                      // alibi slope
        ::testing::Values(-1, 0, 10)                         // sliding window
        ));

}  // namespace llm