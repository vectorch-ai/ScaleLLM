#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "block_allocator.h"
#include "cache_planner.h"

namespace llm {
// GPU physical memory used for key and value cache in attention layers
// the fixed memory is allocated in the constructor for each attention layer
// and is never released.
struct KVCache {
  // the contunuous memory region for key and value cache would be splited into
  // fixed size blocks. the blocks allocation would be managed by the
  // blockallocator.
  torch::Tensor key_cache{};
  torch::Tensor value_cache{};

  // TODO: add functions to access the cache data based on the slot id
};

struct CacheArg {};

// CacheManager is responsible for managing the cache of the LLM model.
class CacheManager final {
 public:
  CacheManager(const CacheArg& cache_arg);

  KVCache& get_kv_cache(uint32_t layer_id) {
    CHECK_LT(layer_id, kv_caches_.size());
    return kv_caches_[layer_id];
  }

 private:
  // actual cache memory allocated for each attention layer
  std::vector<KVCache> kv_caches_;

  // the block allocator that manages the memory blocks mappings.
  // since the memory requirement for each layer are the same, we only need one
  // block allocator to manage the memory blocks for all layers.
  // Question: block_allocator_ should be managed by engine while cache_manager should be part of worker.
  // TODO: revisit the design of block allocator and cache manager
  // std::unique_ptr<BlockAllocator> block_allocator_;
};

}  // namespace llm
