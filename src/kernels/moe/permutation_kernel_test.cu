#include <ATen/ops/allclose.h>
#include <c10/core/ScalarType.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cute/layout.hpp>

#include "gtest/gtest.h"

namespace llm {

namespace kernel::moe {
// forward declare the kernel function
std::tuple<torch::Tensor, torch::Tensor> permute_with_index_map(
    torch::Tensor tokens,  // [n_tokens, dim]
    torch::Tensor indices  // [n_tokens, topk]
);

torch::Tensor unpermute_with_index_map(
    torch::Tensor permuted_tokens,  // [n_permuted_tokens, dim]
    torch::Tensor row_id_map,       // [topk, n_tokens]
    torch::Tensor probs             // [n_tokens, topk]
);

std::tuple<torch::Tensor, torch::Tensor> permute_with_mask_map(
    torch::Tensor tokens,       // [n_tokens, dim]
    torch::Tensor routing_map,  // [n_tokens, n_experts]
    int64_t topk);

torch::Tensor unpermute_with_mask_map(
    torch::Tensor permuted_tokens,  // [n_permuted_tokens, dim]
    torch::Tensor row_id_map,       // [n_experts, n_tokens]
    torch::Tensor probs             // [n_tokens, n_experts]
);

}  // namespace kernel::moe

namespace {
// reference implementation
std::tuple<torch::Tensor, torch::Tensor> permute_index_ref(
    const torch::Tensor& tokens,       // [n_tokens, dim]
    const torch::Tensor& topk_indices  // [n_tokens, topk]
) {
  const auto n_tokens = tokens.size(0);
  const auto topk = topk_indices.size(1);

  auto flatten_indices = topk_indices.view({-1});
  // idx, sorted by (experts, tokens)
  auto sorted_incices = flatten_indices.argsort(/*stable=*/true);

  // idx => token_indices, [n_permuted_tokens]
  auto token_indices = sorted_incices.div(topk, /*rounding_mode=*/"floor");
  auto permuted_tokens = tokens.index_select(
      /*dim=*/0, token_indices);

  return {permuted_tokens, sorted_incices};
}

torch::Tensor unpermute_index_ref(
    const torch::Tensor& permuted_tokens,  // [n_permuted_tokens, dim]
    const torch::Tensor& sorted_incices,   // [n_permuted_tokens]
    const torch::Tensor& probs,            // [n_token, topk]
    int64_t n_tokens,
    int64_t topK) {
  auto tokens = torch::zeros_like(permuted_tokens);

  // [n_permuted_tokens, dim] restore back to original order, sorted by (tokens)
  tokens.index_copy_(
      /*dim=*/0, sorted_incices, permuted_tokens);
  // [n_permuted_tokens, dim] => [n_tokens, topk, dim]
  tokens = tokens.reshape({n_tokens, topK, -1});

  // apply prob
  // [n_tokens, topk, dim] * [n_tokens, topk]
  tokens *= probs.unsqueeze(/*dim=*/-1);

  // [n_tokens, dim], sum over topk
  return tokens.sum(/*dim=*/1);
}

std::tuple<torch::Tensor, torch::Tensor> permute_mask_ref(
    const torch::Tensor& tokens,      // [n_tokens, dim]
    const torch::Tensor& routing_map  // [n_tokens, n_experts]
) {
  const auto n_tokens = routing_map.size(0);
  const auto n_experts = routing_map.size(1);
  const auto options = tokens.options();

  // [n_experts, n_tokens]
  auto token_indices = torch::arange(n_tokens, options.dtype(torch::kLong))
                           .unsqueeze(/*dim=*/0)
                           .expand({n_experts, n_tokens});

  // [n_permuted_tokens] original token indices, sorted by expert idx
  auto sorted_indices = token_indices.masked_select(/*mask=*/routing_map.t());
  auto permuted_tokens = tokens.index_select(
      /*dim=*/0, /*index=*/sorted_indices);
  return {permuted_tokens, sorted_indices};
}

torch::Tensor unpermute_mask_ref(
    const torch::Tensor& permuted_tokens,  // [n_permuted_tokens, dim]
    const torch::Tensor& permuted_probs,   // [n_permuted_tokens]
    const torch::Tensor& sorted_incices,   // [n_permuted_tokens]
    int64_t n_tokens) {
  const auto dim = permuted_tokens.size(1);
  const auto options = permuted_tokens.options();
  // [n_tokens, dim]
  auto tokens = torch::zeros({n_tokens, dim}, options);
  // [n_permuted_tokens] => [n_permuted_tokens, dim]
  auto index = sorted_incices.unsqueeze(/*dim=*/1).expand({-1, dim});
  // reduce sum over experts
  tokens.scatter_add_(
      /*dim=*/0,
      /*index=*/index,
      /*src=*/permuted_tokens * permuted_probs.unsqueeze(/*dim=*/1));
  return tokens;
}

}  // namespace

class PermuteTest
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

TEST_P(PermuteTest, Index) {
  const auto [dtype, n_tokens, dim, n_experts, topk] = GetParam();

  const auto options = torch::dtype(dtype).device(torch::kCUDA);

  const auto tokens = torch::randn({n_tokens, dim}, options);
  const auto gating_logit = torch::randn({n_tokens, n_experts}, options);

  auto [weights, indices] = gating_logit.topk(topk, /*dim=*/-1);
  auto probs = weights.softmax(/*dim=*/-1);

  auto [permuted_tokens, sorted_indices] =
      kernel::moe::permute_with_index_map(tokens, indices.to(torch::kInt32));

  auto [ref_permuted_tokens, ref_sorted_indices] =
      permute_index_ref(tokens, indices);

  EXPECT_TRUE(torch::allclose(permuted_tokens, ref_permuted_tokens));

  auto unpermute_out = kernel::moe::unpermute_with_index_map(
      permuted_tokens, sorted_indices, probs);

  auto ref_unpermute_out = unpermute_index_ref(
      ref_permuted_tokens, ref_sorted_indices, probs, n_tokens, topk);
  EXPECT_TRUE(torch::allclose(
      unpermute_out, ref_unpermute_out, /*rtol=*/1e-2, /*atol=*/1e-2));
  EXPECT_TRUE(
      torch::allclose(tokens, unpermute_out, /*rtol=*/1e-2, /*atol=*/1e-2));
}

TEST_P(PermuteTest, Mask) {
  const auto [dtype, n_tokens, dim, n_experts, topk] = GetParam();

  const auto options = torch::dtype(dtype).device(torch::kCUDA);

  const auto tokens = torch::randn({n_tokens, dim}, options);
  const auto gating_logit = torch::randn({n_tokens, n_experts}, options);

  auto [weights, indices] = gating_logit.topk(topk, /*dim=*/-1);
  //   auto probs = weights.softmax(/*dim=*/-1);

  // construct dense routing map and probs
  auto probs = torch::zeros_like(gating_logit)
                   .scatter(
                       /*dim=*/1, /*index=*/indices, /*value=*/1.0 / topk);
  auto routing_map = torch::zeros_like(gating_logit, torch::kInt)
                         .scatter(
                             /*dim=*/1, /*index=*/indices, /*value=*/1)
                         .to(torch::kBool);

  auto [permuted_tokens, row_id_map] =
      kernel::moe::permute_with_mask_map(tokens, routing_map, topk);

  auto [ref_permuted_tokens, ref_row_id_map] =
      permute_mask_ref(tokens, routing_map);

  EXPECT_TRUE(torch::allclose(permuted_tokens, ref_permuted_tokens));

  auto unpermute_out =
      kernel::moe::unpermute_with_mask_map(permuted_tokens, row_id_map, probs);

  auto ref_permuted_probs = probs.t().masked_select(/*mask=*/routing_map.t());
  auto ref_unpermute_out = unpermute_mask_ref(
      ref_permuted_tokens, ref_permuted_probs, ref_row_id_map, n_tokens);
  EXPECT_TRUE(torch::allclose(
      unpermute_out, ref_unpermute_out, /*rtol=*/1e-2, /*atol=*/1e-2));

  EXPECT_TRUE(torch::allclose(tokens,
                              unpermute_out,
                              /*rtol=*/1e-2,
                              /*atol=*/1e-2));
}

INSTANTIATE_TEST_SUITE_P(
    SM80,
    PermuteTest,
    ::testing::Combine(::testing::Values(torch::kFloat,
                                         torch::kHalf,
                                         torch::kBFloat16),  // dtype
                       ::testing::Values(1, 2, 16),          // n_tokens
                       ::testing::Values(16, 64),            // dim
                       ::testing::Values(4, 8, 16),          // n_experts
                       ::testing::Values(1, 2, 4)            // topk
                       ));

}  // namespace llm
