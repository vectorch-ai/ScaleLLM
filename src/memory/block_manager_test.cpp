#include "block_manager.h"

#include <gtest/gtest.h>

namespace llm {

TEST(BlockManagerTest, Basic) {
  BlockManager::Options options;
  options.num_blocks(10).block_size(2);

  BlockManager manager(options);
  // TODO: add more tests
}

}  // namespace llm