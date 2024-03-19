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

  bool allocate_slots_for_request(Request* request);

  bool allocate_slots_for_sequence(Sequence* sequence);

  bool allocate_slots_for_sequences(std::vector<Sequence*>& sequences);

  void release_slots_for_request(Request* request);

  void release_slots_for_sequences(std::vector<Sequence*>& sequences);

  void release_slots_for_sequence(Sequence* sequence);

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
