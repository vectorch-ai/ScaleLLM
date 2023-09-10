#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include <torch/csrc/distributed/c10d/Backend.hpp>

namespace llm {

class ParallelArgs {
 public:
  ParallelArgs() = default;
  ParallelArgs(int64_t rank, int64_t world_size, c10d::Backend* process_group)
      : rank_(rank), world_size_(world_size), process_group_(process_group) {
    // CHECK_NOTNULL(process_group_);
  }

  // returns pointer to process group
  c10d::Backend* process_group() const { return process_group_; }

  TORCH_ARG(int64_t, rank) = 0;

  TORCH_ARG(int64_t, world_size) = 1;

 private:
  // pointer to process group
  c10d::Backend* process_group_ = nullptr;
};

}  // namespace llm
