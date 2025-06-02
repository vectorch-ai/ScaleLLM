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

torch::Tensor grouped_gemm_ref() {
  return torch::empty({});  // Replace with actual reference computation
}

}  // namespace

class GroupedGemmKernelTest
    : public ::testing::TestWithParam<std::tuple<torch::ScalarType /*q_dtype*/,
                                                 int64_t /*batch_size*/,
                                                 int64_t /*dim*/>> {
 public:
  void SetUp() override {
    // Set random seed for test stability
    torch::manual_seed(0);
  }
};

TEST_P(GroupedGemmKernelTest, GEMM) {
  const auto [dtype, batch_size, dim] = GetParam();

  const auto options = torch::dtype(dtype).device(torch::kCUDA);

  auto ref_out = grouped_gemm_ref();
  auto out = grouped_gemm_sm80();

  EXPECT_TRUE(torch::allclose(out, ref_out, /*rtol=*/1e-3, /*atol=*/1e-3));
}

INSTANTIATE_TEST_SUITE_P(
    GEMM,
    GroupedGemmKernelTest,
    ::testing::Combine(::testing::Values(torch::kHalf),  // q_dtype
                       ::testing::Values(1),             // batch_size
                       ::testing::Values(64)             // dim
                       ));

}  // namespace llm
