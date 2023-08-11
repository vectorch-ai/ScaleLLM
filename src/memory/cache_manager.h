#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "block_allocator.h"
#include "kv_cache.h"

namespace llm {
struct CacheArg {};

// CacheManager is responsible for managing the cache of the LLM model.
class CacheManager final {
 public:
  CacheManager(const CacheArg& cache_arg);

  const KVCache& get_kv_cache(uint32_t layer_id) {
    CHECK_LT(layer_id, kv_caches_.size());
    return kv_caches_[layer_id];
  }

 private:
  // actual cache memory allocated for each attention layer
  std::vector<KVCache> kv_caches_;

  // the block allocator that manages the memory blocks mappings.
  // since the memory requirement for each layer are the same, we only need one
  // block allocator to manage the memory blocks for all layers.
  // Question: block_allocator_ should be managed by engine while cache_manager
  // should be part of worker.
  // TODO: revisit the design of block allocator and cache manager
  // std::unique_ptr<BlockAllocator> block_allocator_;
};

}  // namespace llm
