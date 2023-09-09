#pragma once

#include <torch/torch.h>

namespace llm {

class ParallelArgs {
 public:
  ParallelArgs() = default;
  ParallelArgs(int64_t rank, int64_t world_size)
      : rank_(rank), world_size_(world_size) {}
  
  TORCH_ARG(int64_t, rank) = 0;

  TORCH_ARG(int64_t, world_size) = 1;

  // parameters to create parallel group
};

}  // namespace llm
