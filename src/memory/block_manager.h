#pragma once

#include <cstdint>
#include <vector>

#include "block_allocator.h"
#include "prefix_cache.h"
#include "request/request.h"
#include "request/sequence.h"

namespace llm {

class BlockManager final {
 public:
  BlockManager(uint32_t num_blocks, int32_t block_size);

  bool allocate_blocks_for(Sequence* sequence);

  bool allocate_blocks_for(std::vector<Sequence*>& sequences);

  void release_blocks_for(Request* request);

  void release_blocks_for(std::vector<Sequence*>& sequences);

  void release_blocks_for(Sequence* sequence);

  // try to allocate blloks for sequence with num_tokens
  bool allocate_blocks_for(Sequence* sequence, size_t num_tokens);

 private:
  // check if block allocator has enough slots, if not, try to evict some blocks
  // from the prefix cache
  bool has_enough_blocks(uint32_t num_blocks);

  // try to share blocks among sequences with the same prefix
  void allocate_shared_blocks(Sequence* sequence);

  // number of slots per block
  int32_t block_size_ = 0;

  // the block allocator that manages the memory blocks
  BlockAllocator block_allocator_;

  // prefix cache
  PrefixCache prefix_cache_;
};

}  // namespace llm
