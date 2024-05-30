#include "block_allocator.h"

#include <glog/logging.h>

#include <cstdint>
#include <vector>

#include "block.h"

namespace llm {

BlockAllocator::BlockAllocator(uint32_t total_blocks, uint32_t block_size)
    : num_free_blocks_(total_blocks), block_size_(block_size) {
  CHECK_GT(total_blocks, 0) << "No blocks to allocate";
  CHECK_GT(block_size, 0) << "Block size must be positive";

  free_blocks_.reserve(total_blocks);
  for (int32_t i = 0; i < total_blocks; ++i) {
    // push smaller block ids to the back of the vector
    free_blocks_.push_back(total_blocks - i - 1);
  }
}

BlockAllocator::~BlockAllocator() {
  CHECK(num_free_blocks_ == free_blocks_.size())
      << "Not all blocks have been freed";
}

// allocate a list of block ids
std::vector<Block> BlockAllocator::allocate(uint32_t n_blocks) {
  CHECK(n_blocks <= num_free_blocks_) << "Not enough blocks available";
  std::vector<Block> blocks;
  blocks.reserve(n_blocks);
  for (uint32_t i = 0; i < n_blocks; ++i) {
    const int32_t block_id = free_blocks_[--num_free_blocks_];
    blocks.emplace_back(block_id, this);
  }
  return blocks;
}

// allocate a block id
Block BlockAllocator::allocate() {
  CHECK(num_free_blocks_ > 0) << "No more blocks available";
  const int32_t block_id = free_blocks_[--num_free_blocks_];
  return {block_id, this};
}

// caller should make sure the block_id is valid
void BlockAllocator::free(int32_t block_id) {
  CHECK(num_free_blocks_ < free_blocks_.size());
  free_blocks_[num_free_blocks_++] = block_id;
}

}  // namespace llm
