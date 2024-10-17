#include "memory.h"

#include <c10/cuda/CUDACachingAllocator.h>
#include <glog/logging.h>
#include <torch/torch.h>

namespace llm::memory {

// returns the maximum memory allocated in bytes on the device
// return the peak allocated memory in bytes since the beginning of the
// program.
// Only support CUDA device for now.
int64_t max_memory_allocated(const torch::Device& device) {
  CHECK(device.is_cuda()) << "Only support CUDA device for now.";
  using namespace c10::cuda;
  const auto device_index =
      device.has_index() ? device.index() : current_device();
  const auto stats = CUDACachingAllocator::getDeviceStats(device_index);
  // StatType::AGGREGATE
  return stats.allocated_bytes[0].peak;
}

// returns the total memory in bytes of the device.
// Only support CUDA device for now.
int64_t total_memory(const torch::Device& device) {
  CHECK(device.is_cuda()) << "Only support CUDA device for now.";

  const auto device_index =
      device.has_index() ? device.index() : c10::cuda::current_device();
  cudaDeviceProp prop{};
  const auto err = cudaGetDeviceProperties(&prop, device_index);
  CHECK(err == cudaSuccess) << "Failed to get properties for " << device
                            << ", error: " << cudaGetErrorString(err);
  return static_cast<int64_t>(prop.totalGlobalMem);
}

int64_t available_memory(const torch::Device& device) {
  CHECK(device.is_cuda()) << "Only support CUDA device for now.";
  const auto device_index =
      device.has_index() ? device.index() : c10::cuda::current_device();
  CHECK(cudaSetDevice(device_index) == cudaSuccess)
      << "Failed to set device to " << device_index;
  size_t free = 0;
  size_t total = 0;
  CHECK(cudaMemGetInfo(&free, &total) == cudaSuccess)
      << "Failed to get memory info for " << device;
  return static_cast<int64_t>(free);
}

}  // namespace llm::memory
