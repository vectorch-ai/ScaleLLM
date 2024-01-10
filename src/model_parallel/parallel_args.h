#pragma once

#include <ostream>

#include "common/macros.h"
#include "process_group.h"

namespace llm {

struct ParallelArgs {
  ParallelArgs(int32_t rank, int32_t world_size, ProcessGroup* process_group)
      : rank_(rank), world_size_(world_size), process_group_(process_group) {}

  // rank of current process
  DEFINE_ARG(int32_t, rank) = 0;

  // world size
  DEFINE_ARG(int32_t, world_size) = 0;

  // pointer to process group, nullptr if world size is 1
  DEFINE_PTR_ARG(ProcessGroup, process_group) = nullptr;
};

inline std::ostream& operator<<(std::ostream& os, const ParallelArgs& args) {
  os << "ParallelArgs: [";
  os << "rank: " << args.rank();
  os << ", world_size: " << args.world_size();
  os << "]";
  return os;
}

}  // namespace llm
