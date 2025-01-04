#include <cuda_runtime.h>

#include <cuda/std/chrono>
#include <nvbench/nvbench.cuh>

__global__ void sleep_kernel(nvbench::int64_t microseconds) {
  const auto start = cuda::std::chrono::high_resolution_clock::now();
  const auto target_duration = cuda::std::chrono::microseconds(microseconds);
  const auto finish = start + target_duration;

  auto now = cuda::std::chrono::high_resolution_clock::now();
  while (now < finish) {
    now = cuda::std::chrono::high_resolution_clock::now();
  }
}

void sleep_benchmark(nvbench::state& state) {
  // Collect CUPTI metrics
  state.collect_cupti_metrics();

  // provide reads/writes in bytes
  state.add_global_memory_reads<float>(0);
  state.add_global_memory_writes<float>(0);

  const auto duration_us = state.get_int64("Duration (us)");
  state.exec([&duration_us](nvbench::launch& launch) {
    sleep_kernel<<<1, 1, 0, launch.get_stream()>>>(duration_us);
  });
}

// ::testing::Combine(::testing::Values(1),    // batch_size
//                    ::testing::Values(256),  // q_len
//                    ::testing::Values(256),  // kv_len
//                    ::testing::Values(32),   // n_heads
//                    ::testing::Values(32),   // n_kv_heads
//                    ::testing::Values(64)    // head_dim

NVBENCH_BENCH(sleep_benchmark)
    .add_int64_axis("Duration (us)", nvbench::range(0, 10, 5))
    .set_timeout(1);  // Limit to one second per measurement.