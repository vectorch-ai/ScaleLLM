#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <torch/nn/functional.h>

#include <cstdio>

#include "layernorm_kernels.h"

TEST(NormalizationKernelTest, LayernormFloatTest) {
  float epsilon = 1e-6;
  int m = 32;
  int n = 512;

  auto out = torch::zeros({m, n}, torch::TensorOptions().device(torch::kCUDA));
  auto input =
      torch::randn({m, n}, torch::TensorOptions().device(torch::kCUDA));
  auto weight = torch::randn({n}, torch::TensorOptions().device(torch::kCUDA));
  auto bias = torch::randn({n}, torch::TensorOptions().device(torch::kCUDA));
  auto desired_out = torch::nn::functional::layer_norm(
      input,
      torch::nn::functional::LayerNormFuncOptions({n}).weight(weight).bias(
          bias));

  llm::kernel::layer_norm(out, input, weight, bias, epsilon);

  EXPECT_TRUE(torch::allclose(out, desired_out, 1e-3, 1e-5));
}

TEST(NormalizationKernelTest, LayernormHalfTest) {
  float epsilon = 1e-6;
  int m = 4;
  int n = 512;

  auto out = torch::zeros(
      {m, n},
      torch::TensorOptions().dtype(at::ScalarType::Half).device(torch::kCUDA));
  auto input = torch::randn(
      {m, n},
      torch::TensorOptions().dtype(at::ScalarType::Half).device(torch::kCUDA));
  auto weight = torch::randn(
      {n},
      torch::TensorOptions().dtype(at::ScalarType::Half).device(torch::kCUDA));
  auto bias = torch::randn(
      {n},
      torch::TensorOptions().dtype(at::ScalarType::Half).device(torch::kCUDA));
  auto desired_out = torch::nn::functional::layer_norm(
      input,
      torch::nn::functional::LayerNormFuncOptions({n}).weight(weight).bias(
          bias));

  llm::kernel::layer_norm(out, input, weight, bias, epsilon);

  EXPECT_TRUE(torch::allclose(out, desired_out, 0.05, 1e-3));
}