#include "normalization.h"

#include <ATen/ops/allclose.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch/types.h>

#include "kernels/layernorm_kernels.h"
#include "model_loader/state_dict.h"

namespace llm {

TEST(NormalizationTest, LayerNorm) {
  // TODO: test other device and dtype combinations
  const auto dtype = torch::kFloat;
  const auto device = torch::kCPU;
  const auto options = torch::dtype(dtype).device(device);

  const int64_t dim = 1038;
  const float eps = 1e-5;

  // generate weight
  const auto weight = torch::rand({dim}, options);
  const auto bias = torch::rand({dim}, options);
  StateDict state_dict({{"weight", weight}, {"bias", bias}}, 0, 1);

  LayerNorm norm(dim, eps, /*bias=*/true, options);
  // test load state dict
  norm->load_state_dict(state_dict);
  norm->verify_loaded_weights();

  // verify output
  const auto input = torch::randn({100, dim});
  auto output = norm(input);
  auto desired_output = detail::layer_norm(input, {dim}, weight, bias, eps);
  EXPECT_TRUE(torch::allclose(output, desired_output));
}

TEST(NormalizationTest, LayerNormKernel) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA not available, skipping test";
  }

  // TODO: test other device and dtype combinations
  const auto dtype = torch::kHalf;
  const auto device = torch::kCUDA;
  const auto options = torch::dtype(dtype).device(device);

  const int64_t dim = 1038;
  const float eps = 1e-5;

  // generate weight
  const auto weight = torch::rand({dim}, options);
  const auto bias = torch::rand({dim}, options);

  // verify output
  const auto input = torch::randn({100, dim}, options);

  auto output = torch::empty_like(input);
  kernel::layer_norm(output, input, weight, bias, eps);

  // use float result as baseline
  auto desired_output = detail::layer_norm(input.to(torch::kFloat32),
                                           {dim},
                                           weight.to(torch::kFloat32),
                                           bias.to(torch::kFloat32),
                                           eps)
                            .to(dtype);

  EXPECT_TRUE(torch::allclose(output,
                              desired_output,
                              /*rtol=*/1e-03,
                              /*atol=*/1e-05));
}

TEST(NormalizationTest, RMSNorm) {
  // TODO: test other device and dtype combinations
  const auto dtype = torch::kFloat;
  const auto device = torch::kCPU;
  const auto options = torch::dtype(dtype).device(device);

  const int64_t dim = 1038;
  const float eps = 1e-5;

  // generate weight
  const auto weight = torch::rand({dim}, options);
  StateDict state_dict({{"weight", weight}}, 0, 1);

  RMSNorm norm(dim, eps, options);
  // test load state dict
  norm->load_state_dict(state_dict);
  norm->verify_loaded_weights();

  // verify output
  const auto input = torch::randn({100, dim});
  auto output = norm(input);

  // use float result as baseline
  auto desired_output =
      detail::rms_norm(input.to(torch::kFloat32), weight, eps).to(dtype);
  EXPECT_TRUE(torch::allclose(output, desired_output));
}

TEST(NormalizationTest, RMSNormKernel) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA not available, skipping test";
  }

  // TODO: test other device and dtype combinations
  const auto dtype = torch::kHalf;
  const auto device = torch::kCUDA;
  const auto options = torch::dtype(dtype).device(device);

  const int64_t dim = 1038;
  const float eps = 1e-5;

  // generate weight
  const auto weight = torch::rand({dim}, options);

  // verify output
  const auto input = torch::randn({100, dim}, options);

  auto output = torch::empty_like(input);
  kernel::rms_norm(output, input, weight, eps);

  // use float result as baseline
  auto output_ref = detail::rms_norm(input, weight, eps);

  EXPECT_TRUE(torch::allclose(output,
                              output_ref,
                              /*rtol=*/1e-02,
                              /*atol=*/1e-03));
}

TEST(NormalizationTest, RMSNormResidualKernel) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA not available, skipping test";
  }

  const auto dtype = torch::kHalf;
  const auto device = torch::kCUDA;
  const auto options = torch::dtype(dtype).device(device);

  const int64_t dim = 1024;
  const float eps = 1e-5;

  // generate weight
  const auto weight = torch::rand({dim}, options);

  // verify output
  const auto input = torch::randn({100, dim}, options);
  auto residual = torch::randn({100, dim}, options);
  auto residual_ref = residual.clone();

  auto output = torch::empty_like(input);
  kernel::rms_norm_residual(output, residual, input, weight, eps);

  // use float result as baseline
  auto output_ref = detail::rms_norm_residual(input, residual_ref, weight, eps);

  EXPECT_TRUE(torch::allclose(output,
                              output_ref,
                              /*rtol=*/1e-02,
                              /*atol=*/1e-03));

  EXPECT_TRUE(torch::allclose(residual,
                              residual_ref,
                              /*rtol=*/1e-03,
                              /*atol=*/1e-05));
}

}  // namespace llm
