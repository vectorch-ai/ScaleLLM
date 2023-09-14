#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include <torch/csrc/distributed/c10d/Backend.hpp>

#include "common/arg.h"

namespace llm {

class ParallelArgs {
 public:
  ParallelArgs() = default;

  ParallelArgs(int32_t rank, int32_t world_size, c10d::Backend* process_group)
      : rank_(rank), world_size_(world_size), process_group_(process_group) {
  }

  // rank of current process
  DEFINE_ARG(int32_t, rank) = 0;

  // world size
  DEFINE_ARG(int32_t, world_size) = 0;

    // pointer to process group, nullptr if world size is 1
  DEFINE_PTR_ARG(c10d::Backend, process_group) = nullptr;
};

}  // namespace llm
