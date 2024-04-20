#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <torch/nn/functional.h>

#include <cstdio>

#include "layernorm_kernels.h"

TEST(NormalizationKernelTest, LayernormFloatTest) {
  float epsilon = 1e-6;
  int m = 32;
  int n = 512;

  auto input = torch::randn({m, n});
  auto weight = torch::randn({n});
  auto bias = torch::randn({n});
  auto desired_out = torch::nn::functional::layer_norm(
      input,
      torch::nn::functional::LayerNormFuncOptions({n}).weight(weight).bias(
          bias));

  float* hout = (float*)malloc(m * n * sizeof(float));
  float* hinput = input.data_ptr<float>();
  float* hweight = weight.data_ptr<float>();
  float* hbias = bias.data_ptr<float>();

  float* dout;
  float* dinput;
  float* dweight;
  float* dbias;
  cudaMalloc((void**)&dout, sizeof(float) * m * n);
  cudaMalloc((void**)&dinput, sizeof(float) * m * n);
  cudaMalloc((void**)&dweight, sizeof(float) * n);
  cudaMalloc((void**)&dbias, sizeof(float) * n);

  cudaMemcpy(dinput, hinput, sizeof(float) * m * n, cudaMemcpyHostToDevice);
  cudaMemcpy(dweight, hweight, sizeof(float) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(dbias, hbias, sizeof(float) * n, cudaMemcpyHostToDevice);

  llm::kernel::invoke_layernorm_kernel<float>(
      dout, dinput, dweight, dbias, epsilon, m, n);

  cudaMemcpy(hout, dout, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

  auto out = torch::from_blob(hout, {m, n});
  EXPECT_TRUE(torch::allclose(out, desired_out, 1e-3, 1e-5));
  free(hout);
  cudaFree(dout);
  cudaFree(dinput);
  cudaFree(dweight);
  cudaFree(dbias);
}

TEST(NormalizationKernelTest, LayernormHalfTest) {
  float epsilon = 1e-6;
  int m = 4;
  int n = 512;
  auto input = torch::randn({m, n});
  auto weight = torch::randn({n});
  auto bias = torch::randn({n});
  auto desired_out = torch::nn::functional::layer_norm(
      input,
      torch::nn::functional::LayerNormFuncOptions({n}).weight(weight).bias(
          bias));

  half* hout = (half*)malloc(m * n * sizeof(half));
  half* hinput = (half*)malloc(m * n * sizeof(half));
  half* hweight = (half*)malloc(n * sizeof(half));
  half* hbias = (half*)malloc(n * sizeof(half));

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      hinput[i * n + j] = __float2half(input[i][j].item<float>());
    }
  }
  for (int i = 0; i < weight.numel(); i++)
    hweight[i] = __float2half(weight[i].item<float>());
  for (int i = 0; i < bias.numel(); i++)
    hbias[i] = __float2half(bias[i].item<float>());

  half* dout;
  half* dinput;
  half* dweight;
  half* dbias;
  cudaMalloc((void**)&dout, sizeof(half) * m * n);
  cudaMalloc((void**)&dinput, sizeof(half) * m * n);
  cudaMalloc((void**)&dweight, sizeof(half) * n);
  cudaMalloc((void**)&dbias, sizeof(half) * n);

  cudaMemcpy(dinput, hinput, sizeof(half) * m * n, cudaMemcpyHostToDevice);
  cudaMemcpy(dweight, hweight, sizeof(half) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(dbias, hbias, sizeof(half) * n, cudaMemcpyHostToDevice);

  llm::kernel::invoke_layernorm_kernel<half>(
      dout, dinput, dweight, dbias, epsilon, m, n);

  cudaMemcpy(hout, dout, sizeof(half) * m * n, cudaMemcpyDeviceToHost);

  float* float_hout = (float*)malloc(m * n * sizeof(float));
  for (int i = 0; i < m * n; i++) float_hout[i] = __half2float(hout[i]);

  auto out = torch::from_blob(float_hout, {m, n});
  EXPECT_TRUE(torch::allclose(out, desired_out, 0.05, 1e-3));
  free(hout);
  free(hinput);
  free(hweight);
  free(hbias);
  free(float_hout);
  cudaFree(dout);
  cudaFree(dinput);
  cudaFree(dweight);
  cudaFree(dbias);
}
