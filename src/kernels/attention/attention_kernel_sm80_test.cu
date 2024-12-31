#include <gtest/gtest.h>
#include <torch/torch.h>

#include "attention_kernel_sm80.cuh"

namespace llm {
namespace {
// Multi-head attention implementation using pytorch
torch::Tensor attention_ref(
    torch::Tensor query,  // [q_seq_len, n_heads, head_dim]
    torch::Tensor key,    // [seq_len, n_kv_heads, head_dim]
    torch::Tensor value   // [seq_len, n_kv_heads, head_dim]
) {
  const auto q_seq_len = query.size(0);
  const auto n_heads = query.size(1);
  const auto head_dim = query.size(2);
  const auto seq_len = key.size(0);
  const auto n_kv_heads = key.size(1);

  assert(n_heads == n_kv_heads);

  // query * key => [n_heads, q_seq_len, seq_len]
  auto scores = torch::einsum("qhd,khd->hqk", {query, key});
  // apply scale
  const float sm_scale = static_cast<float>(1.0 / std::sqrt(head_dim));
  scores *= sm_scale;

  // safe softmax
  scores = torch::softmax(scores, /*dim=*/-1);

  // score * value => [q_seq_len, n_heads, head_dim]
  return torch::einsum("hqk,khd->qhd", {scores, value});
}

}  // namespace

class AttentionKernelTest
    : public ::testing::TestWithParam<std::tuple<int64_t /*seq_len*/,
                                                 int64_t /*q_seq_len*/,
                                                 int64_t /*n_heads*/,
                                                 int64_t /*n_kv_heads*/,
                                                 int64_t /*head_dim*/>> {};

TEST_P(AttentionKernelTest, MHA) {
  const auto [seq_len, q_seq_len, n_heads, n_kv_heads, head_dim] = GetParam();

  const auto options = torch::dtype(torch::kFloat).device(torch::kCPU);

  const auto query = torch::randn({q_seq_len, n_heads, head_dim}, options);
  const auto key = torch::randn({seq_len, n_kv_heads, head_dim}, options);
  const auto value = torch::randn({seq_len, n_kv_heads, head_dim}, options);

  auto ref_out = attention_ref(query, key, value);

  // auto out = torch::empty_like(query);
  // mha(query, key, value, out);
  // EXPECT_TRUE(torch::allclose(out, ref_out, /*rtol=*/1e-5, /*atol=*/1e-5));
}

INSTANTIATE_TEST_SUITE_P(MHA,
                         AttentionKernelTest,
                         ::testing::Combine(::testing::Values(64),  // seq_len
                                            ::testing::Values(64),  // q_seq_len
                                            ::testing::Values(8),   // n_heads
                                            ::testing::Values(8),  // n_kv_heads
                                            ::testing::Values(64)  // head_dim
                                            ));

}  // namespace llm