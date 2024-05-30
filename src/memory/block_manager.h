#pragma once

#include <cstdint>
#include <vector>

#include "block_allocator.h"
#include "common/macros.h"
#include "memory/block.h"
#include "prefix_cache.h"
#include "request/request.h"
#include "request/sequence.h"

namespace llm {

class BlockManager final {
 public:
  struct Options {
    DEFINE_ARG(uint32_t, num_blocks) = 0;

    DEFINE_ARG(int32_t, block_size) = 0;

    DEFINE_ARG(bool, enable_prefix_cache) = true;
  };

  BlockManager(const Options& options);

  bool allocate_blocks_for(Sequence* sequence);

  bool allocate_blocks_for(std::vector<Sequence*>& sequences);

  void release_blocks_for(Request* request);

  void release_blocks_for(std::vector<Sequence*>& sequences);

  void release_blocks_for(Sequence* sequence);

  // try to allocate blloks for sequence with num_tokens
  bool allocate_blocks_for(Sequence* sequence, size_t num_tokens);

  // try to share blocks among sequences with the same prefix
  void allocate_shared_blocks_for(Sequence* sequence);

  // cache the blocks for the sequence
  void cache_blocks_for(Sequence* sequence);

  // get the options for the block manager
  const Options& options() const { return options_; }

  // get the number of blocks in the prefix cache
  size_t num_blocks_in_prefix_cache() const {
    return prefix_cache_.num_blocks();
  }

  // get the number of free blocks in the block allocator
  size_t num_free_blocks() const { return block_allocator_.num_free_blocks(); }

  // get the effective number of blocks in use
  size_t num_blocks_in_use() const { return num_blocks_in_use_; }

  // get the block utilization.
  double kv_cache_utilization() const {
    return static_cast<double>(num_blocks_in_use_) /
           block_allocator_.num_total_blocks();
  }

 private:
  // check if block allocator has enough slots, if not, try to evict some blocks
  // from the prefix cache
  bool has_enough_blocks(uint32_t num_blocks);

  // the options for the block manager
  Options options_;

  // the block allocator that manages the memory blocks
  BlockAllocator block_allocator_;

  // prefix cache
  PrefixCache prefix_cache_;

  // reserved block id for padding
  Block padding_block_;

  // number of blocks in use
  size_t num_blocks_in_use_ = 0;
};

}  // namespace llm
