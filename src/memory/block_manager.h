#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "block_allocator.h"

namespace llm {
class Request;

class BlockManager final {
 public:
  BlockManager(uint32_t num_blocks, uint32_t block_size);

  // try to allocat slots for the request
  bool allocate_slots_for_request(Request* request);

  // preempt a request to release allocated blocks for other high priority
  // requests.
  void release_slots_for_request(Request* request);

 private:
  // number of slots per block
  uint32_t slots_per_block_ = 0;

  // the block allocator that manages the memory blocks
  BlockAllocator block_allocator_;
};

}  // namespace llm
