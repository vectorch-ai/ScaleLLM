#include "block_manager.h"

#include <glog/logging.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "block_allocator.h"
#include "request/request.h"

namespace llm {
BlockManager::BlockManager(uint32_t num_blocks, uint32_t slots_per_block)
    : slots_per_block_(slots_per_block),
      block_allocator_(num_blocks, slots_per_block) {}

// try to allocat slots for the request
bool BlockManager::allocate_slots_for_request(Request* request) {
  DCHECK(request != nullptr);
  uint32_t num_blocks_to_allocate = 0;
  for (const auto& sequence : request->sequences) {
    num_blocks_to_allocate += sequence.num_blocks_to_allocate(slots_per_block_);
  }

  if (num_blocks_to_allocate > block_allocator_.free_block_count()) {
    // not enough blocks
    return false;
  }
  for (auto& sequence : request->sequences) {
    const uint32_t blocks_to_allocate =
        sequence.num_blocks_to_allocate(slots_per_block_);
    const auto block_ids = block_allocator_.allocate(blocks_to_allocate);
    sequence.add_blocks(block_ids);
  }
  return true;
}

// preempt a request to release allocated blocks for other high priority
// requests.
void BlockManager::release_slots_for_request(Request* request) {
  DCHECK(request != nullptr);
  for (auto& sequence : request->sequences) {
    const auto block_ids = sequence.release_blocks();
    // add block ids back to the block allocator
    block_allocator_.free(block_ids);
  }
}

}  // namespace llm
