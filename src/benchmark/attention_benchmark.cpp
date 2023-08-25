#include <benchmark/benchmark.h>

#include "layers/attention.h"

static void BM_varlen_masked_self_attention(benchmark::State& state) {
  // Perform setup here

  for (auto _ : state) {
    // Call the implementation function
  }
}

static void BM_single_token_masked_self_attention(benchmark::State& state) {
  // Perform setup here

  for (auto _ : state) {
    // Call the implementation function
  }
}

// Register functions as benchmarks
BENCHMARK(BM_varlen_masked_self_attention)
    ->Args({/*dim=*/32, /*device=cpu*/0})
    ->Args({/*dim=*/32, /*device=cuda*/1, /*slow=*/0})
    ->Args({/*dim=*/32, /*device=cuda*/1, /*slow=*/1});

BENCHMARK(BM_single_token_masked_self_attention)
    ->Args({/*dim=*/32, /*device=cpu*/ 0})
    ->Args({/*dim=*/32, /*device=cuda*/ 1});
