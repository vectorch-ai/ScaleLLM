#include "block_allocator.h"
#include "block.h"

#include <glog/logging.h>

#include <cstdint>
#include <vector>

namespace llm {

BlockAllocator::BlockAllocator(uint32_t total_blocks, uint32_t block_size)
      : free_block_count_(total_blocks), block_size_(block_size) {
    free_blocks_.reserve(free_block_count_);
    for (int32_t i = 0; i < free_block_count_; ++i) {
      // push smaller block ids to the back of the vector
      free_blocks_.push_back(free_block_count_ - i - 1);
    }
  }

  // allocate a list of block ids
  std::vector<Block> BlockAllocator::allocate(uint32_t n_blocks) {
    std::vector<Block> blocks;
    blocks.reserve(n_blocks);
    for (uint32_t i = 0; i < n_blocks; ++i) {
      blocks.push_back(allocate());
    }
    return blocks;
  }

  // allocate a block id
  Block BlockAllocator::allocate() {
    CHECK(free_block_count_ > 0) << "No more blocks available";
    int32_t block_id = free_blocks_[--free_block_count_];
    return {block_id, this};
  }

  // caller should make sure the block_id is valid
  void BlockAllocator::free(int32_t block_id) {
    CHECK(free_block_count_ < free_blocks_.size());
    free_blocks_[free_block_count_++] = block_id;
  }

}  // namespace llm
