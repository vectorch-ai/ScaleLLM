#include "prefix_cache.h"

#include <gtest/gtest.h>

namespace llm {

TEST(PrefixCacheTest, Basic) {
  const uint32_t block_size = 2;
  PrefixCache cache(block_size);

  // Test match with empty cache
  {
    std::vector<int32_t> token_ids = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    std::vector<Block> blocks = cache.match(token_ids);
    EXPECT_EQ(blocks.size(), 0);
  }

  // Test insert three sequences
  //   tokens: [1, 2] -> [5, 6, 7, 8, 9, 10]*
  //                  -> [3, 4] -> [5, 6]*
  //                            -> [50, 60, 70, 80, 90, 100]*
  //   blocks: [0] -> [5, 15, 25]*
  //               -> [1] -> [2]*
  //                      -> [20, 30, 40]*
  {
    // insert sequence: [1, 2, 3, 4, 5, 6]
    std::vector<int32_t> token_ids = {1, 2, 3, 4, 5, 6, 7};
    std::vector<Block> blocks = {0, 1, 2};
    uint32_t len = cache.insert(token_ids, blocks);
    // truncate at block boundary
    EXPECT_EQ(len, 6);
    EXPECT_EQ(cache.num_blocks(), 3);  // [0, 1, 2]
    EXPECT_EQ(cache.num_nodes(), 1);

    // insert sequence: [1, 2, 3, 4] -> new [50, 60, 70, 80, 90, 100]
    // expected two sequences split at [1, 2, 3, 4]
    //    tokens: [1, 2, 3, 4] -> [5, 6]*
    //                         -> [50, 60, 70, 80, 90, 100]*
    //    blocks: [0, 1] -> [2]*
    //                   -> [20, 30, 40]*
    token_ids = {1, 2, 3, 4, 50, 60, 70, 80, 90, 100, 110};
    blocks = {0, 1, 20, 30, 40, 50};
    len = cache.insert(token_ids, blocks);
    // truncate at block boundary
    EXPECT_EQ(len, 6);                 // [50, 60, 70, 80, 90, 100]
    EXPECT_EQ(cache.num_blocks(), 6);  // [0, 1] -> [2] | [20, 30, 40]
    EXPECT_EQ(cache.num_nodes(), 3);

    // insert sequence [1, 2, 5, 6, 7, 8, 9, 10]
    // expect 3 sequences split at [1, 2]
    //   tokens: [1, 2] -> [5, 6, 7, 8, 9, 10]*
    //                  -> [3, 4] -> [5, 6]*
    //                            -> [50, 60, 70, 80, 90, 100]*
    //   blocks: [0] -> [5, 15, 25]*
    //               -> [1] -> [2]*
    //                      -> [20, 30, 40]*
    token_ids = {1, 2, 5, 6, 7, 8, 9, 10, 11};
    blocks = {0, 5, 15, 25, 35};
    len = cache.insert(token_ids, blocks);
    // truncate at block boundary
    EXPECT_EQ(len, 6);  // [5, 6, 7, 8, 9, 10]
    EXPECT_EQ(cache.num_blocks(), 9);
    EXPECT_EQ(cache.num_nodes(), 5);
  }

  // Test match with cache:
  //   tokens: [1, 2] -> [5, 6, 7, 8, 9, 10]*
  //                  -> [3, 4] -> [5, 6]*
  //                            -> [50, 60, 70, 80, 90, 100]*
  //   blocks: [0] -> [5, 15, 25]*
  //               -> [1] -> [2]*
  //                      -> [20, 30, 40]*
  {
    // no match
    std::vector<int32_t> token_ids = {3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<Block> blocks = cache.match(token_ids);
    EXPECT_TRUE(blocks.empty());

    // match first sequence partially
    token_ids = {1, 2, 5, 6, 8};
    blocks = cache.match(token_ids);
    std::vector<Block> desired_blocks = {0, 5};
    EXPECT_EQ(blocks, desired_blocks);

    // match second sequence fully
    token_ids = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    blocks = cache.match(token_ids);
    desired_blocks = {0, 1, 2};
    EXPECT_EQ(blocks, desired_blocks);

    // match third sequence partially
    token_ids = {1, 2, 3, 4, 50, 60, 70, 80, 90};
    blocks = cache.match(token_ids);
    desired_blocks = {0, 1, 20, 30};
    EXPECT_EQ(blocks, desired_blocks);
  }

  // Test evict
  //   tokens: [1, 2] -> [5, 6, 7, 8, 9, 10]*
  //                  -> [3, 4] -> [5, 6]*
  //                            -> [50, 60, 70, 80, 90, 100]*
  //   blocks: [0] -> [5, 15, 25]*
  //               -> [1] -> [2]*
  //                      -> [20, 30, 40]*
  {
    // Hold sequence to prevent evicting
    std::vector<int32_t> token_ids = {1, 2, 5, 6};
    std::vector<Block> blocks = cache.match(token_ids);
    std::vector<Block> desired_blocks = {0, 5};
    EXPECT_EQ(blocks, desired_blocks);

    // evict 2 blocks to test partial eviction
    uint32_t evicted = cache.evict(2);
    EXPECT_EQ(evicted, 2);
    EXPECT_EQ(cache.num_blocks(), 7);

    // try to evict all blocks, ending with 2 hold blocks left
    const size_t total_blocks = cache.num_blocks();
    evicted = cache.evict(total_blocks);
    EXPECT_EQ(evicted, 5);
    EXPECT_EQ(cache.num_blocks(), 2);
    EXPECT_EQ(cache.num_nodes(), 2);

    // release blocks then evict all
    blocks.clear();
    evicted = cache.evict(total_blocks);
    EXPECT_EQ(evicted, 2);
    EXPECT_EQ(cache.num_blocks(), 0);
    EXPECT_EQ(cache.num_nodes(), 0);
  }
}

class PrefixCacheRandomTest
    : public ::testing::TestWithParam<
          std::tuple<int32_t /*block_size*/, int32_t /*max_seq_len*/>> {};

TEST_P(PrefixCacheRandomTest, Random) {
  const auto& [block_size, max_seq_len] = GetParam();
  // sample which existing sequence to share the prefix
  // sample the length of the prefix
  // generate the sequence and blocks
  // insert the sequence and blocks into prefix cache
  // match the sequence and blocks from the prefix cache
  // save the sequence and blocks into a vector

  // randomly query the prefix cache and compare the result with the saved vector
}

INSTANTIATE_TEST_SUITE_P(
    Random,
    PrefixCacheRandomTest,
    ::testing::Combine(::testing::Values(1, 2, 4, 48, 256),  // block_size
                       ::testing::Values(10, 1000)           // max_seq_len
                       ));

}  // namespace llm