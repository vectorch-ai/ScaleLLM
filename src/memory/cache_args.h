#pragma once

#include <torch/torch.h>

#include <cstddef>
#include <optional>

#include "common/arg.h"

namespace llm {

inline constexpr int64_t GB = int64_t(1024) * 1024 * 1024;

class CacheArgs {
 public:
  // number of slots per block, each slot is for one token
  DEFINE_ARG(int32_t, block_size) = 0;

  // number of blocks in the cache
  DEFINE_ARG(int64_t, num_blocks) = 0;

  // maximum memory utilization allowed
  DEFINE_ARG(double, max_memory_utilization) = 0.9;

  // maximum cache size in bytes
  DEFINE_ARG(int64_t, max_cache_size) = 4 * GB;
};

}  // namespace llm
