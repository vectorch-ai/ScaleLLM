#include <ATen/core/TensorBody.h>
#include <c10/core/ScalarType.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cute/layout.hpp>

namespace llm {
namespace kernel {
// forward declare the kernel function
void grouped_topk_sigmoid(
    const torch::Tensor& gating_logits,    // [n_tokens, n_experts]
    const torch::Tensor& correction_bias,  // [n_experts]
    const int n_expert_groups,
    const int topk_group,
    const int topk,
    float scaling_factor,
    torch::Tensor& topk_weights,  // [n_tokens, topk]
    torch::Tensor& topk_indices   // [n_tokens, topk]
);
}  // namespace kernel

namespace {

std::tuple<torch::Tensor, torch::Tensor> grouped_topk_sigmoid_ref(
    const torch::Tensor& gating_logits,    // [n_tokens, n_experts]
    const torch::Tensor& correction_bias,  // [n_experts]
    float scaling_factor,
    int64_t n_expert_groups,
    int64_t topk_group,
    int64_t topk) {
  const auto n_tokens = gating_logits.size(0);
  const auto n_experts = gating_logits.size(1);
  assert(n_experts % n_expert_groups == 0);
  const auto group_size = n_experts / n_expert_groups;

  // [n_tokens, n_experts] original scores
  auto scores = gating_logits.sigmoid();

  // select topk_group groups
  // [n_tokens, n_experts]
  auto scores_for_choice = scores + correction_bias.unsqueeze(/*dim=*/0);
  // [n_tokens, n_groups]
  auto group_scores =
      std::get<0>(scores_for_choice.view({n_tokens, n_expert_groups, -1})
                      .topk(2, /*dim=*/-1))
          .sum(/*dim=*/-1);
  // [n_tokens, topk_group]
  auto [group_weights, group_indices] = group_scores.topk(
      topk_group, /*dim=*/-1, /*largest=*/true, /*sorted=*/false);
  // [n_tokens, topk_group]
  auto group_mask = torch::zeros_like(group_scores);
  group_mask.scatter_(
      /*dim=*/-1,
      group_indices,
      /*src=*/1.0f);
  // [n_tokens, n_experts]
  auto score_mask = group_mask.unsqueeze(/*dim=*/-1)
                        .expand(
                            /*size=*/{n_tokens, n_expert_groups, group_size})
                        .reshape({n_tokens, -1});
  // [n_tokens, n_experts]
  auto tmp_scores = scores_for_choice.masked_fill(
      /*mask=*/score_mask == 0,
      /*value=*/-std::numeric_limits<float>::infinity());
  // [n_tokens, topk]
  auto [topk_weights, topk_indices] =
      tmp_scores.topk(topk, /*dim=*/1, /*largest=*/true, /*sorted=*/false);
  // fetch the original scores
  // [n_tokens, topk]
  topk_weights = scores.gather(/*dim=*/1, topk_indices);

  // apply the scaling factor
  topk_weights = topk_weights * scaling_factor;
  return {topk_weights, topk_indices};
}

}  // namespace

class GroupedTopkSigmoidTest
    : public ::testing::TestWithParam<std::tuple<torch::ScalarType /*q_dtype*/,
                                                 int64_t /*n_tokens*/,
                                                 int64_t /*n_experts*/,
                                                 int64_t /*n_expert_groups*/,
                                                 int64_t /*topk_group*/,
                                                 int64_t /*topk*/>> {
 public:
  void SetUp() override {
    // Set random seed for test stability
    torch::manual_seed(0);
  }
};

TEST_P(GroupedTopkSigmoidTest, TopkSoftmax) {
  const auto [dtype, n_tokens, n_experts, n_expert_groups, topk_group, topk] =
      GetParam();

  const auto options = torch::dtype(dtype).device(torch::kCUDA);

  const auto gating_logits = torch::randn({n_tokens, n_experts}, options);
  const auto correction_bias = torch::randn({n_experts}, options);
  const auto scaling_factor = 1.0f;

  // output
  auto weights = torch::empty({n_tokens, topk}, options);
  auto indices = torch::empty({n_tokens, topk}, options.dtype(torch::kInt32));
  // kernel
  kernel::grouped_topk_sigmoid(gating_logits,
                               correction_bias,
                               n_expert_groups,
                               topk_group,
                               topk,
                               scaling_factor,
                               weights,
                               indices);

  // reference
  auto [ref_weights, ref_indices] = grouped_topk_sigmoid_ref(gating_logits,
                                                             correction_bias,
                                                             scaling_factor,
                                                             n_expert_groups,
                                                             topk_group,
                                                             topk);

  EXPECT_TRUE(torch::allclose(weights, ref_weights));
  EXPECT_TRUE(torch::equal(indices, ref_indices.to(torch::kInt32)));
}

INSTANTIATE_TEST_SUITE_P(
    Moe,
    GroupedTopkSigmoidTest,
    ::testing::Combine(::testing::Values(torch::kFloat),  // dtype
                       ::testing::Values(10),             // n_tokens
                       ::testing::Values(128, 256),       // n_experts
                       ::testing::Values(8),              // n_expert_groups
                       ::testing::Values(4),              // topk_group
                       ::testing::Values(1, 2)            // topk
                       ));

}  // namespace llm