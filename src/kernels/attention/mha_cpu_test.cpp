#include "mha_cpu.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

namespace llm {
namespace {
// Multi-head attention implementation using pytorch
torch::Tensor masked_self_attention(
    torch::Tensor query,  // [q_seq_len, n_heads, head_dim]
    torch::Tensor key,    // [seq_len, n_kv_heads, head_dim]
    torch::Tensor value   // [seq_len, n_kv_heads, head_dim]
) {
  const auto q_seq_len = query.size(0);
  const auto n_heads = query.size(1);
  const auto head_dim = query.size(2);
  const auto seq_len = key.size(0);
  const auto n_kv_heads = key.size(1);

  // repeat key and value if n_kv_heads < n_heads
  if (n_kv_heads < n_heads) {
    assert(n_heads % n_kv_heads == 0);
    const auto n_groups = n_heads / n_kv_heads;
    key = key.repeat_interleave(/*repeats=*/n_groups, /*dim=*/1);
    value = value.repeat_interleave(/*repeats=*/n_groups, /*dim=*/1);
  }

  // query * key => [n_heads, q_seq_len, seq_len]
  auto scores = torch::einsum("qhd,khd->hqk", {query, key});
  // apply scale
  const float sm_scale = static_cast<float>(1.0 / std::sqrt(head_dim));
  scores *= sm_scale;

  // apply causal mask
  torch::Tensor mask = torch::ones({1, q_seq_len, seq_len}, torch::kBool);
  // returns the lower triangular part of a matrix
  mask = torch::tril(mask, /*diagonal=*/seq_len - q_seq_len).to(query);
  scores = scores.masked_fill(mask == 0, -5e4);

  // safe softmax
  scores = torch::softmax(scores, /*dim=*/-1);

  // score * value => [q_seq_len, n_heads, head_dim]
  return torch::einsum("hqk,khd->qhd", {scores, value});
}

}  // namespace

class AttentionCPUTest
    : public ::testing::TestWithParam<std::tuple<int64_t /*seq_len*/,
                                                 int64_t /*q_seq_len*/,
                                                 int64_t /*n_heads*/,
                                                 int64_t /*n_kv_heads*/,
                                                 int64_t /*head_dim*/>> {
 public:
  void SetUp() override {
    // Set random seed for test stability
    torch::manual_seed(0);
  }
};

TEST_P(AttentionCPUTest, MHA) {
  const auto [seq_len, q_seq_len, n_heads, n_kv_heads, head_dim] = GetParam();

  const auto options = torch::dtype(torch::kFloat).device(torch::kCPU);

  const auto query = torch::randn({q_seq_len, n_heads, head_dim}, options);
  const auto key = torch::randn({seq_len, n_kv_heads, head_dim}, options);
  const auto value = torch::randn({seq_len, n_kv_heads, head_dim}, options);

  auto ref_out = masked_self_attention(query, key, value);

  auto out = torch::empty_like(query);
  mha(query, key, value, out);
  EXPECT_TRUE(torch::allclose(out, ref_out, /*rtol=*/1e-4, /*atol=*/1e-4));
}

INSTANTIATE_TEST_SUITE_P(
    MHA,
    AttentionCPUTest,
    ::testing::Combine(::testing::Values(20),       // seq_len
                       ::testing::Values(10, 20),   // q_seq_len
                       ::testing::Values(6),        // n_heads
                       ::testing::Values(6, 3, 1),  // n_kv_heads
                       ::testing::Values(32)        // head_dim
                       ));

}  // namespace llm