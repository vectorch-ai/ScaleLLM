#include "normalization.h"

#include <ATen/ops/allclose.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include "model_loader/state_dict.h"

namespace llm {
namespace {

torch::Tensor layer_norm(torch::Tensor input,
                         const torch::Tensor& weight,
                         const torch::Tensor& bias,
                         double eps) {
  const auto mean = input.mean(/*dim=*/-1, /*keepdim=*/true);
  const auto variance = input.var(/*dim=*/-1,
                                  /*unbiased=*/false,
                                  /*keepdim=*/true);
  auto norm = (input - mean) / torch::sqrt(variance + eps);
  norm *= weight;
  if (bias.defined()) {
    norm += bias;
  }
  return norm;
}
}  // namespace

TEST(NormalizationTest, LayerNorm) {
  // TODO: test other device and dtype combinations
  const auto dtype = torch::kFloat;
  const auto device = torch::kCPU;

  const int64_t dim = 10;
  const double eps = 1e-5;

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
  auto desired_output = layer_norm(input, weight, bias, eps);
  EXPECT_TRUE(torch::allclose(output,
                              desired_output,
                              /*rtol=*/1e-02,
                              /*atol=*/1e-03));
}

}  // namespace llm
