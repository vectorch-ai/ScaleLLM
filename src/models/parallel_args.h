#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include <torch/csrc/distributed/c10d/Backend.hpp>

namespace llm {

class ParallelArgs {
 public:
  ParallelArgs(int32_t rank, int32_t world_size, c10d::Backend* process_group)
      : rank_(rank), world_size_(world_size), process_group_(process_group) {
    if (world_size_ > 1) {
      CHECK_NOTNULL(process_group_);
    }
  }

  // returns pointer to process group
  c10d::Backend* process_group() const { return process_group_; }

  // returns rank of current process
  int32_t rank() const { return rank_; }

  // returns world size
  int32_t world_size() const { return world_size_; }

 private:
  // rank of current process
  int32_t rank_ = 0;
  // world size
  int32_t world_size_ = 1;

  // pointer to process group, not null if world_size > 1
  c10d::Backend* process_group_ = nullptr;
};

}  // namespace llm
