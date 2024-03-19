#include "block_manager.h"

#include <gtest/gtest.h>

namespace llm {

TEST(BlockManagerTest, Basic) {
  const uint32_t n_blocks = 10;
  const uint32_t block_size = 2;
  BlockManager manager(n_blocks, block_size);
  // TODO: add more tests
}

}  // namespace llm