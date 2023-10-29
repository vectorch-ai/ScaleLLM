#include "memory.h"

#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/torch.h>

#include "common/logging.h"

namespace llm::memory {

// returns the maximum memory allocated in bytes on the device
// return the peak allocated memory in bytes since the beginning of the
// program.
// Only support CUDA device for now.
int64_t max_memory_allocated(const torch::Device& device) {
  GCHECK(device.is_cuda()) << "Only support CUDA device for now.";
  using namespace c10::cuda;
  const auto device_index =
      device.has_index() ? device.index() : current_device();
  const auto stats = CUDACachingAllocator::getDeviceStats(device_index);
  return stats
      .allocated_bytes[static_cast<size_t>(
          CUDACachingAllocator::StatType::AGGREGATE)]
      .peak;
}

// returns the total memory in bytes of the device.
// Only support CUDA device for now.
int64_t total_memory(const torch::Device& device) {
  GCHECK(device.is_cuda()) << "Only support CUDA device for now.";

  const auto device_index =
      device.has_index() ? device.index() : c10::cuda::current_device();
  cudaDeviceProp prop{};
  const auto err = cudaGetDeviceProperties(&prop, device_index);
  GCHECK(err == cudaSuccess) << "Failed to get properties for " << device
                             << ", error: " << cudaGetErrorString(err);
  return static_cast<int64_t>(prop.totalGlobalMem);
}

}  // namespace llm::memory
