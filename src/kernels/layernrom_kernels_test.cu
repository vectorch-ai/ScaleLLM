#include <cuda_fp16.h>

#include <cstdio>

#include "layernorm_kernels.h"

template <typename T>
void printMatrix(T* a, int m, int n) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      printf("%f ", (float)a[i * n + j]);
    }
    puts("");
  }
  puts("");
}

template <>
void printMatrix<half2>(half2* a, int m, int n) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      printf(
          "%f %f ", __half2float(a[i * n + j].x), __half2float(a[i * n + j].y));
    }
    puts("");
  }
  puts("");
}

void layernorm_kernel_half2_test() {
  float epsilon = 1e-6;
  int m = 2;
  int n = 2;

  half2* out = (half2*)malloc(m * n * sizeof(half2));
  half2* input = (half2*)malloc(m * n * sizeof(half2));
  half2* weight = (half2*)malloc(m * n * sizeof(half2));
  half2* bias = (half2*)malloc(m * n * sizeof(half2));

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      input[i * n + j] = half2(__float2half((float)(i * n + j * 2)),
                               __float2half((float)(i * n + j * 2 + 1)));
      weight[i * n + j] = half2(__float2half(1.), __float2half(1.));
      bias[i * n + j] = half2(__float2half(0.), __float2half(0.));
    }
  }

  half2* dout;
  half2* dinput;
  half2* dweight;
  half2* dbias;
  cudaMalloc((void**)&dout, sizeof(half2) * m * n);
  cudaMalloc((void**)&dinput, sizeof(half2) * m * n);
  cudaMalloc((void**)&dweight, sizeof(half2) * m * n);
  cudaMalloc((void**)&dbias, sizeof(half2) * m * n);

  cudaMemcpy(dinput, input, sizeof(half2) * m * n, cudaMemcpyHostToDevice);
  cudaMemcpy(dweight, weight, sizeof(half2) * m * n, cudaMemcpyHostToDevice);
  cudaMemcpy(dbias, bias, sizeof(half2) * m * n, cudaMemcpyHostToDevice);

  llm::kernel::invoke_layernorm_kernel<half2>(
      dout, dinput, dweight, dbias, epsilon, m, n);

  cudaMemcpy(out, dout, sizeof(half2) * m * n, cudaMemcpyDeviceToHost);

  printf("---------- test half2 layernorm kernel -----------\n");
  printf("input:\n");
  printMatrix<half2>(input, m, n);
  printf("weights:\n");
  printMatrix<half2>(weight, m, n);
  printf("bias:\n");
  printMatrix<half2>(bias, m, n);
  printf("outputs:\n");
  printMatrix<half2>(out, m, n);
}

void layernorm_kernel_float_test() {
  float epsilon = 1e-6;
  int m = 2;
  int n = 4;

  float* out = (float*)malloc(m * n * sizeof(float));
  float* input = (float*)malloc(m * n * sizeof(float));
  float* weight = (float*)malloc(m * n * sizeof(float));
  float* bias = (float*)malloc(m * n * sizeof(float));

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      input[i * n + j] = (float)(i * n + j);
      weight[i * n + j] = 1.;
      bias[i * n + j] = 0.;
    }
  }

  float* dout;
  float* dinput;
  float* dweight;
  float* dbias;
  cudaMalloc((void**)&dout, sizeof(float) * m * n);
  cudaMalloc((void**)&dinput, sizeof(float) * m * n);
  cudaMalloc((void**)&dweight, sizeof(float) * m * n);
  cudaMalloc((void**)&dbias, sizeof(float) * m * n);

  cudaMemcpy(dinput, input, sizeof(float) * m * n, cudaMemcpyHostToDevice);
  cudaMemcpy(dweight, weight, sizeof(float) * m * n, cudaMemcpyHostToDevice);
  cudaMemcpy(dbias, bias, sizeof(float) * m * n, cudaMemcpyHostToDevice);

  llm::kernel::invoke_layernorm_kernel<float>(
      dout, dinput, dweight, dbias, epsilon, m, n);

  cudaMemcpy(out, dout, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

  printf("---------- test float layernorm kernel -----------\n");
  printf("input:\n");
  printMatrix<float>(input, m, n);
  printf("weights:\n");
  printMatrix<float>(weight, m, n);
  printf("bias:\n");
  printMatrix<float>(bias, m, n);
  printf("outputs:\n");
  printMatrix<float>(out, m, n);
}

int main() {
  layernorm_kernel_float_test();
  layernorm_kernel_half2_test();
  return 0;
}