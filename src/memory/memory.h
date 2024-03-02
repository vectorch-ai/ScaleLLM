#pragma once
#include <torch/torch.h>

namespace llm::memory {

// returns the maximum memory allocated in bytes on the device
// return the peak allocated memory in bytes since the beginning of the
// program.
// Only support CUDA device for now.
int64_t max_memory_allocated(const torch::Device& device);

// returns the total memory in bytes of the device.
// Only support CUDA device for now.
int64_t total_memory(const torch::Device& device);

// returns the available memory in bytes of the device.
int64_t available_memory(const torch::Device& device);

} // namespace llm::memory
