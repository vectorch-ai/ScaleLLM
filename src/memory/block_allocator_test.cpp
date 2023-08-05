#include "block_allocator.h"

#include <gtest/gtest.h>

namespace llm {

TEST(BlockAllocatorTest, Basic) {
  const uint32_t num_cpu_blocks = 20;
  const uint32_t num_gpu_blocks = 10;
  const uint32_t block_size_in_bytes = 1024;
  BlockAllocator allocator(num_cpu_blocks, num_gpu_blocks, block_size_in_bytes);

  EXPECT_EQ(allocator.free_cpu_block_count(), num_cpu_blocks);
  EXPECT_EQ(allocator.free_gpu_block_count(), num_gpu_blocks);
  EXPECT_EQ(allocator.block_size_in_bytes(), block_size_in_bytes);

  // Allocate a CPU block
  {
    Block block = allocator.allocate(MemoryType::kCPU);
    EXPECT_EQ(block.id(), 0);
    EXPECT_EQ(block.size(), block_size_in_bytes);
    EXPECT_EQ(block.memory_type(), MemoryType::kCPU);
    EXPECT_EQ(block.is_shared(), false);
    EXPECT_EQ(block.ref_count(), 1);

    EXPECT_EQ(allocator.free_cpu_block_count(), num_cpu_blocks - 1);
    EXPECT_EQ(allocator.free_gpu_block_count(), num_gpu_blocks);
  }
  // the block should be freed after the scope
  EXPECT_EQ(allocator.free_cpu_block_count(), num_cpu_blocks);
  EXPECT_EQ(allocator.free_gpu_block_count(), num_gpu_blocks);

  // Allocate a list of gpu blocks
  {
    std::vector<Block> blocks;
    for (uint32_t i = 0; i < num_gpu_blocks; ++i) {
      blocks.push_back(allocator.allocate(MemoryType::kGPU));
      EXPECT_EQ(blocks.back().id(), i);
      EXPECT_EQ(blocks.back().size(), block_size_in_bytes);
      EXPECT_EQ(blocks.back().memory_type(), MemoryType::kGPU);
      EXPECT_EQ(blocks.back().is_shared(), false);
      EXPECT_EQ(blocks.back().ref_count(), 1);
    }
    EXPECT_EQ(allocator.free_cpu_block_count(), num_cpu_blocks);
    EXPECT_EQ(allocator.free_gpu_block_count(), 0);
    for (const auto& block : blocks) {
      EXPECT_EQ(block.ref_count(), 1);
      EXPECT_EQ(block.is_shared(), false);
    }

    // Test CHECK failure
    EXPECT_DEATH(allocator.allocate(MemoryType::kGPU),
                 "No more GPU memory blocks available");
  }
  // all blocks should be freed after the scope
  EXPECT_EQ(allocator.free_cpu_block_count(), num_cpu_blocks);
  EXPECT_EQ(allocator.free_gpu_block_count(), num_gpu_blocks);

  // Test shared blocks
  {
    Block block = allocator.allocate(MemoryType::kCPU);
    EXPECT_EQ(block.ref_count(), 1);
    EXPECT_EQ(block.is_shared(), false);
    // test copy constructor
    {
      // NOLINTNEXTLINE
      const Block block2 = block;
      EXPECT_EQ(block.ref_count(), 2);
      EXPECT_EQ(block.is_shared(), true);
      EXPECT_EQ(block2.ref_count(), 2);
      EXPECT_EQ(block2.is_shared(), true);
    }
    EXPECT_EQ(block.ref_count(), 1);
    EXPECT_EQ(block.is_shared(), false);

    // test move constructor
    {
      Block block3 = std::move(block);
      EXPECT_EQ(block3.ref_count(), 1);
      EXPECT_EQ(block3.is_shared(), false);

      EXPECT_EQ(block.ref_count(), 0);
    }
    EXPECT_EQ(block.ref_count(), 0);
  }
}

}  // namespace llm
