#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>

#include "grouped_gemm_kernel_sm80.cuh"  // IWYU pragma: keep

namespace llm {

namespace {

torch::Tensor grouped_gemm_sm80() {
  auto out = torch::empty({});  // Placeholder for output tensor

  using Traits = GEMMTraitsSM80<cute::half_t, /*DTYPE*/
                                64,           /*DIM*/
                                64,           /*BLK_M*/
                                64,           /*BLK_N*/
                                64,           /*BLK_K*/
                                2>;           /*STAGES*/

  GEMMParams params;
  launch_grouped_gemm_kernel_sm80<Traits>(params, nullptr);

  return out;
}

// returns (m, topk, n)
torch::Tensor grouped_gemm_ref(const torch::Tensor& a,        // (m, k)
                               const torch::Tensor& w,        // (e, n, k)
                               const torch::Tensor& topk_ids  // (m, topk)

) {
  const auto m = a.size(0);
  const auto k = a.size(1);
  const auto n = w.size(1);
  const auto n_experts = w.size(0);
  const auto topk = topk_ids.size(1);

  // (m * topk, n)
  auto out = torch::zeros({m * topk, n}, a.options());

  // (m, k) => (m, topk, k) => (m * topk, k)
  auto a_expanded_flat =
      a.unsqueeze(/*dim=*/1).expand({-1, topk, -1}).reshape({-1, k});
  // (m, topk) => (m * topk)
  auto topk_ids_flat = topk_ids.reshape({-1});

  // process each expert
  for (int64_t e = 0; e < n_experts; ++e) {
    // 1D indices for the current expert
    auto indices = torch::nonzero(topk_ids_flat == e).squeeze();
    // select corresponding tokens
    auto a_selected = a_expanded_flat.index_select(/*dim=*/0, indices);
    // perform the GEMM operation for this expert
    auto e_out = torch::matmul(a_selected, w[e].transpose(0, 1));
    // copy the results into the output tensor
    out.index_copy_(/*dim=*/0, indices, e_out);
  }
  // (m * topk, n) => (m, topk, n)
  return out.view({m, topk, n});
}

}  // namespace

class GroupedGemmKernelTest
    : public ::testing::TestWithParam<std::tuple<torch::ScalarType /*dtype*/,
                                                 int64_t /*m*/,
                                                 int64_t /*n*/,
                                                 int64_t /*k*/,
                                                 int64_t /*n_experts*/,
                                                 int64_t /*topk*/>> {
 public:
  void SetUp() override {
    // Set random seed for test stability
    torch::manual_seed(0);
  }
};

TEST_P(GroupedGemmKernelTest, GEMM) {
  const auto [dtype, m, n, k, n_experts, topk] = GetParam();

  const auto options = torch::dtype(dtype).device(torch::kCUDA);

  // Create input tensors
  auto a = torch::randn({m, k}, options);
  auto w = torch::randn({n_experts, n, k}, options);

  // Get top-k indices
  auto logits = torch::randn({m, n_experts}, options).softmax(/*dim=*/1);
  auto [topk_weights, topk_ids] = logits.topk(topk, /*dim=*/1);

  // LOG(ERROR) << "a: " << a;
  // LOG(ERROR) << "w: " << w;
  // LOG(ERROR) << "topk_ids: " << topk_ids;
  // LOG(ERROR) << "topk_weights: " << topk_weights;

  auto ref_out = grouped_gemm_ref(a, w, topk_ids);
  // LOG(ERROR) << "ref_out: " << ref_out;
  // auto out = grouped_gemm_sm80();

  // EXPECT_TRUE(torch::allclose(out, ref_out, /*rtol=*/1e-3, /*atol=*/1e-3));
}

INSTANTIATE_TEST_SUITE_P(
    GEMM,
    GroupedGemmKernelTest,
    ::testing::Combine(::testing::Values(torch::kHalf),  // dtype
                       ::testing::Values(1),             // m
                       ::testing::Values(64),            // n
                       ::testing::Values(64),            // k
                       ::testing::Values(8),             // n_experts
                       ::testing::Values(4)              // topk
                       ));

}  // namespace llm
