#include <ATen/ops/allclose.h>
#include <c10/core/ScalarType.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cute/layout.hpp>

#include "gtest/gtest.h"

namespace llm {

namespace kernel::moe {
void permute_align_block(torch::Tensor topk_ids,
                         int64_t num_experts,
                         int64_t block_size,
                         torch::Tensor sorted_token_ids,
                         torch::Tensor experts_ids,
                         torch::Tensor num_tokens_post_pad,
                         torch::Tensor cumsum_buffer);

}  // namespace kernel::moe

namespace {
// reference implementation
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> permute_align_block_ref(
    torch::Tensor topk_ids,
    int64_t n_experts,
    int64_t block_size) {
  //   return {sorted_token_ids, experts_ids, num_tokens_post_pad};
  return {topk_ids, topk_ids, topk_ids};
}

}  // namespace

class AlignBlockTest
    : public ::testing::TestWithParam<std::tuple<torch::ScalarType /*q_dtype*/,
                                                 int64_t /*n_tokens*/,
                                                 int64_t /*dim*/,
                                                 int64_t /*n_experts*/,
                                                 int64_t /*topk*/>> {
 public:
  void SetUp() override {
    // Set random seed for test stability
    torch::manual_seed(0);
  }
};

TEST_P(AlignBlockTest, AlignBlock) {
  const auto [dtype, n_tokens, dim, n_experts, topk] = GetParam();

  const auto options = torch::dtype(dtype).device(torch::kCUDA);
  const auto gating_logit = torch::randn({n_tokens, n_experts}, options);

  auto [weights, indices] = gating_logit.topk(topk, /*dim=*/-1);
  auto probs = weights.softmax(/*dim=*/-1);

  //   auto [permuted_tokens, sorted_indices] =
  //       kernel::moe::permute_with_index_map(tokens,
  //       indices.to(torch::kInt32));

  //   auto [ref_permuted_tokens, ref_sorted_indices] =
  //       permute_index_ref(tokens, indices);

  //   EXPECT_TRUE(torch::allclose(permuted_tokens, ref_permuted_tokens));

  //   auto unpermute_out = kernel::moe::unpermute_with_index_map(
  //       permuted_tokens, sorted_indices, probs);

  //   auto ref_unpermute_out = unpermute_index_ref(
  //       ref_permuted_tokens, ref_sorted_indices, probs, n_tokens, topk);
  //   EXPECT_TRUE(torch::allclose(
  //       unpermute_out, ref_unpermute_out, /*rtol=*/1e-2, /*atol=*/1e-2));
  //   EXPECT_TRUE(
  //       torch::allclose(tokens, unpermute_out, /*rtol=*/1e-2,
  //       /*atol=*/1e-2));
}

INSTANTIATE_TEST_SUITE_P(
    Moe,
    AlignBlockTest,
    ::testing::Combine(::testing::Values(torch::kFloat,
                                         torch::kHalf,
                                         torch::kBFloat16),  // dtype
                       ::testing::Values(1, 2, 16),          // n_tokens
                       ::testing::Values(16, 64),            // dim
                       ::testing::Values(4, 8, 16),          // n_experts
                       ::testing::Values(1, 2, 4)            // topk
                       ));

}  // namespace llm
