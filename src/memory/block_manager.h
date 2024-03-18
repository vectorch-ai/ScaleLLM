#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "block_allocator.h"
#include "request/request.h"

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
  // number of slots per block
  int32_t block_size_ = 0;

  // the block allocator that manages the memory blocks
  BlockAllocator block_allocator_;
};

}  // namespace llm
