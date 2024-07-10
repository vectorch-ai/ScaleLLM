#include <nvbench/nvbench.cuh>

#include <cuda/std/chrono>

#include <cuda_runtime.h>

__global__ void sleep_kernel(nvbench::int64_t microseconds) {
  const auto start = cuda::std::chrono::high_resolution_clock::now();
  const auto target_duration = cuda::std::chrono::microseconds(microseconds);
  const auto finish = start + target_duration;

  auto now = cuda::std::chrono::high_resolution_clock::now();
  while (now < finish) {
    now = cuda::std::chrono::high_resolution_clock::now();
  }
}

void sleep_benchmark(nvbench::state &state) {
  const auto duration_us = state.get_int64("Duration (us)");
  state.exec([&duration_us](nvbench::launch &launch) {
    sleep_kernel<<<1, 1, 0, launch.get_stream()>>>(duration_us);
  });
}
NVBENCH_BENCH(sleep_benchmark)
    .add_int64_axis("Duration (us)", nvbench::range(0, 100, 5))
    .set_timeout(1); // Limit to one second per measurement.