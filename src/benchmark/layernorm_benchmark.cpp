#include <benchmark/benchmark.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <cuda_runtime.h>

#include "kernels/layernorm_kernels.h"
#include "layers/normalization.h"

using namespace llm;
using namespace llm::detail;

static void BM_rms_norm(benchmark::State& state,
                          const torch::Device& device) {
  // skip if no gpu
  if (device.is_cuda() && !torch::cuda::is_available()) {
    state.SkipWithMessage("CUDA is not available");
    return;
  }

  // Perform setup here
  torch::ScalarType dtype = static_cast<torch::ScalarType>(state.range(0));
  std::string normalization = "rms_norm";
  auto normalization_func = rms_norm;
  int64_t dim_0 = state.range(1);
  int64_t dim_1 = state.range(2);
  auto input = torch::rand({dim_0, dim_1}, torch::dtype(dtype).device(device));
  auto weight = torch::ones({dim_0, dim_1}, torch::dtype(dtype).device(device));
  float epsilon = 1e-5;
  for (auto _ : state) {
    // Call the implementation function
    auto output = normalization_func(input, weight, epsilon);
    // don't optimize out the output
    benchmark::DoNotOptimize(output);
  }
  state.SetLabel(normalization + " " + torch::toString(dtype));
}

static void BM_layer_norm(benchmark::State& state,
                          const torch::Device& device) {
  // skip if no gpu
  if (device.is_cuda() && !torch::cuda::is_available()) {
    state.SkipWithMessage("CUDA is not available");
    return;
  }
  // Perform setup here
  torch::ScalarType dtype = static_cast<torch::ScalarType>(state.range(0));
  std::string normalization = "layer_norm";
  auto normalization_func = layer_norm;
  int64_t dim_0 = state.range(1);
  int64_t dim_1 = state.range(2);
  auto input = torch::rand({dim_0, dim_1}, torch::dtype(dtype).device(device));
  std::vector<int64_t> normalized_shape = {dim_1};
  auto weight = torch::ones({dim_1}, torch::dtype(dtype).device(device));
  auto bias = torch::rand({dim_1}, torch::dtype(dtype).device(device));
  float epsilon = 1e-5;
  for (auto _ : state) {
    // Call the implementation function
    auto output = normalization_func(input, normalized_shape, weight, bias, epsilon);
    // auto output = normalization_func(input, weight, epsilon);
    // don't optimize out the output
    benchmark::DoNotOptimize(output);
  }
  state.SetLabel(normalization + " " + torch::toString(dtype));
}

static void BM_rms_norm_kernel(benchmark::State& state) {
  // skip if no gpu
  if (!torch::cuda::is_available()) {
    state.SkipWithMessage("CUDA is not available");
    return;
  }

  // Perform setup here
  torch::ScalarType dtype = static_cast<torch::ScalarType>(state.range(0));
  std::string activation = "rms_norm";
  auto activation_func = kernel::rms_norm;
  int64_t dim_0 = state.range(1);
  int64_t dim_1 = state.range(2);

  auto output = torch::rand({dim_0, dim_1}, torch::dtype(dtype).device(torch::kCUDA));
  auto input = torch::rand({dim_0, dim_1}, torch::dtype(dtype).device(torch::kCUDA));
  auto weight = torch::ones({dim_0, dim_1}, torch::dtype(dtype).device(torch::kCUDA));
  float epsilon = 1e-5;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (auto _ : state) {
    // Start measuring time
    cudaEventRecord(start);

    // Launch the CUDA kernel
    activation_func(output, input, weight, epsilon);
    // don't optimize out the output
    benchmark::DoNotOptimize(output);

    // Stop measuring time and calculate the elapsed time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    state.PauseTiming();

    // Update the benchmark state with the measured time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    state.SetIterationTime(milliseconds / 1000);
    state.ResumeTiming();
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  state.SetLabel(activation + " " + torch::toString(dtype));
}

static void BM_layer_norm_kernel(benchmark::State& state) {
  // skip if no gpu
  if (!torch::cuda::is_available()) {
    state.SkipWithMessage("CUDA is not available");
    return;
  }

  // Perform setup here
  torch::ScalarType dtype = static_cast<torch::ScalarType>(state.range(0));
  std::string activation = "layer_norm";
  auto activation_func = kernel::layer_norm;
  int64_t dim_0 = state.range(1);
  int64_t dim_1 = state.range(2);

  auto output = torch::rand({dim_0, dim_1}, torch::dtype(dtype).device(torch::kCUDA));
  auto input = torch::rand({dim_0, dim_1}, torch::dtype(dtype).device(torch::kCUDA));
  std::vector<int64_t> normalized_shape = {dim_1};
  auto weight = torch::ones({dim_1}, torch::dtype(dtype).device(torch::kCUDA));
  auto bias = torch::rand({dim_1}, torch::dtype(dtype).device(torch::kCUDA));
  float epsilon = 1e-5;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (auto _ : state) {
    // Start measuring time
    cudaEventRecord(start);

    // Launch the CUDA kernel
    activation_func(output, input, weight, bias, epsilon);
    // don't optimize out the output
    benchmark::DoNotOptimize(output);

    // Stop measuring time and calculate the elapsed time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    state.PauseTiming();

    // Update the benchmark state with the measured time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    state.SetIterationTime(milliseconds / 1000);
    state.ResumeTiming();
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  state.SetLabel(activation + " " + torch::toString(dtype));
}

// Register functions as benchmarks
const std::vector<int64_t> dtypes = {static_cast<int64_t>(torch::kFloat),
                                     static_cast<int64_t>(torch::kHalf),
                                     static_cast<int64_t>(torch::kBFloat16)};

// benchmark for kernels
BENCHMARK(BM_rms_norm_kernel)
    ->ArgsProduct({dtypes, {4096, 1000}, {20560, 1024}});

BENCHMARK(BM_layer_norm_kernel)
    ->ArgsProduct({dtypes, {4096, 1000}, {20560, 1024}});

// benchmark for gpus
BENCHMARK_CAPTURE(BM_rms_norm, "gpu", torch::kCUDA)
    ->ArgsProduct({dtypes, {4096, 1000}, {20560, 1024}});

BENCHMARK_CAPTURE(BM_layer_norm, "gpu", torch::kCUDA)
    ->ArgsProduct({dtypes, {4096, 1000}, {20560, 1024}});

// benchmark for cpus
BENCHMARK_CAPTURE(BM_rms_norm, "cpu", torch::kCPU)
    ->ArgsProduct({{static_cast<int64_t>(torch::kFloat)},
                   {4096, 1000},
                   {20560, 1024}});
BENCHMARK_CAPTURE(BM_layer_norm, "cpu", torch::kCPU)
    ->ArgsProduct({{static_cast<int64_t>(torch::kFloat)},
                   {4096, 1000},
                   {20560, 1024}});