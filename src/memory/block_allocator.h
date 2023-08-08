#pragma once

#include <glog/logging.h>

#include <cstdint>
#include <vector>

namespace llm {

// BlockAllocator is used to track memory blocks. It is not thread safe.
// Please note: The actual memory has been allocated outside of this class. This
// class only manages the allocation and deallocation of block ids.
class BlockAllocator final {
 public:
  // block_size: number of slots per block
  BlockAllocator(uint32_t num_blocks, uint32_t slots_per_block)
      : free_block_count_(num_blocks), slots_per_block_(slots_per_block) {
    free_blocks_.reserve(free_block_count_);
    for (uint32_t i = 0; i < free_block_count_; ++i) {
      // push smaller block ids to the back of the vector
      free_blocks_.push_back(free_block_count_ - i - 1);
    }
  }

  // allocate a list of block ids
  std::vector<uint32_t> allocate(uint32_t num_blocks) {
    std::vector<uint32_t> block_ids;
    block_ids.reserve(num_blocks);
    for (uint32_t i = 0; i < num_blocks; ++i) {
      block_ids.push_back(allocate());
    }
    return block_ids;
  }

  // free a list of block ids
  void free(const std::vector<uint32_t>& block_ids) {
    for (uint32_t block_id : block_ids) {
      free(block_id);
    }
  }

  // allocate a block id
  uint32_t allocate() {
    CHECK(free_block_count_ > 0) << "No more CPU memory blocks available";
    return free_blocks_[--free_block_count_];
  }

  // caller should make sure the block_id is valid
  void free(uint32_t block_id) {
    CHECK(free_block_count_ < free_blocks_.size());
    free_blocks_[free_block_count_++] = block_id;
  }

  // get number of slots per block
  uint32_t slots_per_block() const { return slots_per_block_; }

  // get number of free blocks
  uint32_t free_block_count() const { return free_block_count_; }

 private:
  // free block count
  uint32_t free_block_count_ = 0;

  // number of slots per block
  uint32_t slots_per_block_ = 0;

  // free block list
  std::vector<uint32_t> free_blocks_;
};

}  // namespace llm
