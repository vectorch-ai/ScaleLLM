#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include "common/arg.h"
#include "common/process_group.h"

namespace llm {

class ParallelArgs {
 public:
  ParallelArgs() = default;

  ParallelArgs(int32_t rank, int32_t world_size, ProcessGroup* process_group)
      : rank_(rank), world_size_(world_size), process_group_(process_group) {
  }

  // rank of current process
  DEFINE_ARG(int32_t, rank) = 0;

  // world size
  DEFINE_ARG(int32_t, world_size) = 0;

    // pointer to process group, nullptr if world size is 1
  DEFINE_PTR_ARG(ProcessGroup, process_group) = nullptr;
};

}  // namespace llm
