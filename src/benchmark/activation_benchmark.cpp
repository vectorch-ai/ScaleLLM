#include <benchmark/benchmark.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>

#include "kernels/activation_kernels.h"
#include "layers/activation.h"

using namespace llm;
using namespace llm::detail;
const std::vector<std::tuple<std::string, ActFunc>> activations = {
    {"gelu", gelu},
    {"gelu_fast", gelu_fast},
    {"gelu_new", gelu_new},
    {"gelu_pytorch_tanh", gelu_pytorch_tanh},
    {"relu", relu},
    {"silu", silu},
};
static void BM_activation(benchmark::State& state,
                          const torch::Device& device) {
  // skip if no gpu
  if (device.is_cuda() && !torch::cuda::is_available()) {
    state.SkipWithMessage("CUDA is not available");
    return;
  }

  // Perform setup here
  torch::ScalarType dtype = static_cast<torch::ScalarType>(state.range(0));
  auto [activation, activation_func] = activations[state.range(1)];
  int64_t dim_0 = state.range(2);
  int64_t dim_1 = state.range(3);
  auto input = torch::rand({dim_0, dim_1}, torch::dtype(dtype).device(device));
  for (auto _ : state) {
    // Call the implementation function
    auto output = activation_func(input);
    // don't optimize out the output
    benchmark::DoNotOptimize(output);
  }
  state.SetLabel(activation + " " + torch::toString(dtype));
}

const std::vector<std::tuple<std::string, ActFunc>> activation_kernels = {
    {"gelu_fast", kernel::gelu_fast},
    {"gelu_new", kernel::gelu_new},
    {"silu", kernel::silu},
};

static void BM_activation_kernel(benchmark::State& state) {
  // skip if no gpu
  if (!torch::cuda::is_available()) {
    state.SkipWithMessage("CUDA is not available");
    return;
  }

  // Perform setup here
  torch::ScalarType dtype = static_cast<torch::ScalarType>(state.range(0));
  auto [activation, activation_func] = activation_kernels[state.range(1)];
  int64_t dim_0 = state.range(2);
  int64_t dim_1 = state.range(3);

  auto input =
      torch::rand({dim_0, dim_1}, torch::dtype(dtype).device(torch::kCUDA));

  for (auto _ : state) {
    // Call the implementation function
    auto output = activation_func(input);
    // don't optimize out the output
    benchmark::DoNotOptimize(output);
  }

  state.SetLabel(activation + " " + torch::toString(dtype));
}

// Register functions as benchmarks
// benchmark for cpus
BENCHMARK_CAPTURE(BM_activation, "cpu", torch::kCPU)
    ->ArgsProduct({{static_cast<int64_t>(torch::kFloat)},
                   {0, 1, 2, 3, 4, 5},
                   {4096, 1000},
                   {20560, 1024}});

// benchmark for gpus
const std::vector<int64_t> dtypes = {static_cast<int64_t>(torch::kFloat),
                                     static_cast<int64_t>(torch::kHalf),
                                     static_cast<int64_t>(torch::kBFloat16)};
BENCHMARK_CAPTURE(BM_activation, "gpu", torch::kCUDA)
    ->ArgsProduct({dtypes, {0, 1, 2, 3, 4, 5}, {4096, 1000}, {20560, 1024}});

// benchmark for kernels
BENCHMARK(BM_activation_kernel)
    ->ArgsProduct({dtypes, {0, 1, 2}, {4096, 1000}, {20560, 1024}});
