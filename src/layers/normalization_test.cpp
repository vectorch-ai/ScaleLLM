#include "normalization.h"

#include <ATen/ops/allclose.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include "kernels/layernorm_kernels.h"
#include "model_loader/state_dict.h"

namespace llm {

TEST(NormalizationTest, LayerNorm) {
  // TODO: test other device and dtype combinations
  const auto dtype = torch::kFloat;
  const auto device = torch::kCPU;

  const int64_t dim = 1038;
  const float eps = 1e-5;

  // generate weight
  const auto weight = torch::rand({dim}, torch::dtype(dtype).device(device));
  const auto bias = torch::rand({dim}, torch::dtype(dtype).device(device));
  StateDict state_dict({{"weight", weight}, {"bias", bias}}, 0, 1);

  LayerNorm norm(dim, eps, /*bias=*/true, dtype, device);
  // test load state dict
  norm->load_state_dict(state_dict);
  norm->verify_loaded_weights();

  // verify output
  const auto input = torch::randn({100, dim});
  auto output = norm(input);
  auto desired_output = detail::layer_norm(input, {dim}, weight, bias, eps);
  EXPECT_TRUE(torch::allclose(output,
                              desired_output,
                              /*rtol=*/1e-02,
                              /*atol=*/1e-03));
}

TEST(NormalizationTest, LayerNormKernel) {
  // TODO: test other device and dtype combinations
  const auto dtype = torch::kHalf;
  const auto device = torch::kCUDA;

  const int64_t dim = 1038;
  const float eps = 1e-5;

  // generate weight
  const auto weight = torch::rand({dim}, torch::dtype(dtype).device(device));
  const auto bias = torch::rand({dim}, torch::dtype(dtype).device(device));

  // verify output
  const auto input =
      torch::randn({100, dim}, torch::dtype(dtype).device(device));

  auto output = torch::empty_like(input);
  kernel::layer_norm(output, input, weight, bias, eps);

  auto desired_output = detail::layer_norm(input, {dim}, weight, bias, eps);

  EXPECT_TRUE(torch::allclose(output,
                              desired_output,
                              /*rtol=*/1e-02,
                              /*atol=*/1e-03));
}


TEST(NormalizationTest, RMSNorm) {
  // TODO: test other device and dtype combinations
  const auto dtype = torch::kFloat;
  const auto device = torch::kCPU;

  const int64_t dim = 1038;
  const float eps = 1e-5;

  // generate weight
  const auto weight = torch::rand({dim}, torch::dtype(dtype).device(device));
  StateDict state_dict({{"weight", weight}}, 0, 1);

  RMSNorm norm(dim, eps, dtype, device);
  // test load state dict
  norm->load_state_dict(state_dict);
  norm->verify_loaded_weights();

  // verify output
  const auto input = torch::randn({100, dim});
  auto output = norm(input);
  auto desired_output = detail::rms_norm(input, weight, eps);
  EXPECT_TRUE(torch::allclose(output,
                              desired_output,
                              /*rtol=*/1e-02,
                              /*atol=*/1e-03));
}

TEST(NormalizationTest, RMSNormKernel) {
  // TODO: test other device and dtype combinations
  const auto dtype = torch::kHalf;
  const auto device = torch::kCUDA;

  const int64_t dim = 1038;
  const float eps = 1e-5;

  // generate weight
  const auto weight = torch::rand({dim}, torch::dtype(dtype).device(device));

  // verify output
  const auto input =
      torch::randn({100, dim}, torch::dtype(dtype).device(device));

  auto output = torch::empty_like(input);
  kernel::rms_norm(output, input, weight, eps);

  auto desired_output = detail::rms_norm(input, weight, eps);

  EXPECT_TRUE(torch::allclose(output,
                              desired_output,
                              /*rtol=*/1e-02,
                              /*atol=*/1e-03));
}

}  // namespace llm
