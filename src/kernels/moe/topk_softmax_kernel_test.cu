#include <c10/core/ScalarType.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cute/layout.hpp>

namespace llm {
namespace kernel {
// forward declare the kernel function
void topk_softmax(const torch::Tensor& gating_logit,  // [n_tokens, n_experts]
                  torch::Tensor& topk_weights,        // [n_tokens, topk]
                  torch::Tensor& topk_indices         // [n_tokens, topk]
);
}  // namespace kernel

class TopkSoftmaxTest
    : public ::testing::TestWithParam<std::tuple<torch::ScalarType /*q_dtype*/,
                                                 int64_t /*n_tokens*/,
                                                 int64_t /*n_experts*/,
                                                 int64_t /*topk*/>> {
 public:
  void SetUp() override {
    // Set random seed for test stability
    torch::manual_seed(0);
  }
};

TEST_P(TopkSoftmaxTest, TopkSoftmax) {
  const auto [dtype, n_tokens, n_experts, topk] = GetParam();

  const auto options = torch::dtype(dtype).device(torch::kCUDA);

  const auto gating_logit = torch::randn({n_tokens, n_experts}, options);

  // output
  auto weights = torch::empty({n_tokens, topk}, options);
  auto indices = torch::empty({n_tokens, topk}, options.dtype(torch::kInt32));
  // kernel
  kernel::topk_softmax(gating_logit, weights, indices);

  // reference
  auto [ref_weights, ref_indices] =
      gating_logit.softmax(/*dim=*/-1).topk(topk, /*dim=*/-1);

  EXPECT_TRUE(torch::allclose(weights, ref_weights));
  EXPECT_TRUE(torch::equal(indices, ref_indices.to(torch::kInt32)));
}

INSTANTIATE_TEST_SUITE_P(
    Moe,
    TopkSoftmaxTest,
    ::testing::Combine(::testing::Values(torch::kFloat),  // q_dtype
                       ::testing::Values(10),             // n_tokens
                       ::testing::Values(8, 16),          // n_experts
                       ::testing::Values(1, 2, 4)         // topk
                       ));

}  // namespace llm