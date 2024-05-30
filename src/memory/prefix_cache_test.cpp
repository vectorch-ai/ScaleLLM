#include "prefix_cache.h"

#include <absl/random/random.h>
#include <gtest/gtest.h>

#include "block_allocator.h"

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

struct SequenceData {
  std::vector<int32_t> token_ids;
  std::vector<Block> blocks;
};

// get a sub vector
template <typename T>
std::vector<T> sub_vector(const std::vector<T>& data, size_t size) {
  return {data.begin(), data.begin() + size};
}

class PrefixCacheRandomTest
    : public ::testing::TestWithParam<std::tuple<int32_t /*block_size*/,
                                                 int32_t /*max_seq_len*/,
                                                 int32_t /*num_seqs*/>> {};

TEST_P(PrefixCacheRandomTest, Random) {
  const auto& [block_size, max_seq_len, num_seqs] = GetParam();

  const int32_t vocab_size = 2000;
  const int32_t total_blocks = (max_seq_len * num_seqs) / block_size + 10;

  BlockAllocator allocator(total_blocks, block_size);
  PrefixCache cache(block_size);

  absl::BitGen gen;
  // construct sequences and insert into prefix cache
  std::vector<SequenceData> seqs_data;
  for (int i = 0; i < num_seqs; i++) {
    {
      // generate random sequence
      // which seq to get common prefix, seq_idx == -1, no common prefix
      int32_t seq_idx = i == 0 ? -1 : absl::Uniform<int32_t>(gen, 0, num_seqs);

      // generate token ids
      std::vector<int32_t> token_ids;
      std::vector<Block> blocks;
      int32_t prefix_len = 0;
      // get common prefix
      if (seq_idx < seqs_data.size()) {
        // common prefix len
        prefix_len =
            absl::Uniform<int32_t>(gen, 0, seqs_data[seq_idx].token_ids.size());
        token_ids = sub_vector(seqs_data[seq_idx].token_ids, prefix_len);
      }
      // total seq len
      int32_t seq_len = absl::Uniform<int32_t>(gen, prefix_len, max_seq_len);
      // generate rest of the sequence
      for (size_t j = token_ids.size(); j < seq_len; j++) {
        token_ids.push_back(absl::Uniform<int32_t>(gen, 0, vocab_size));
      }

      // get shared blocks from prefix cache
      blocks = cache.match(token_ids);

      // allocate blocks for rest of the sequence
      size_t num_blocks = (seq_len + block_size - 1) / block_size;
      for (size_t j = blocks.size(); j < num_blocks; j++) {
        blocks.push_back(allocator.allocate());
      }

      // insert the sequence and blocks into prefix cache
      cache.insert(token_ids, blocks);

      size_t cached_len = seq_len / block_size;
      blocks.resize(cached_len);

      // query back and check
      std::vector<Block> matched_blocks = cache.match(token_ids);
      EXPECT_EQ(matched_blocks, blocks);

      // save the sequence and blocks
      seqs_data.push_back({token_ids, blocks});
    }

    // all blocks either in cache or allocator
    ASSERT_EQ(cache.num_blocks() + allocator.num_free_blocks(), total_blocks);
  }

  // randomly query the prefix cache and compare the result with the saved
  for (int i = 0; i < 1000; i++) {
    const int32_t seq_idx = absl::Uniform<int32_t>(gen, 0, num_seqs);
    const int32_t seq_len =
        absl::Uniform<int32_t>(gen, 0, seqs_data[seq_idx].token_ids.size());

    // randomly generate partial sequence
    std::vector<int32_t> token_ids =
        sub_vector(seqs_data[seq_idx].token_ids, seq_len);
    std::vector<Block> desired_blocks =
        sub_vector(seqs_data[seq_idx].blocks, seq_len / block_size);

    // match the sequence and compare the result
    std::vector<Block> blocks = cache.match(token_ids);
    EXPECT_EQ(blocks, desired_blocks);
  }

  // can't evict any blocks since all blocks hold by seqs_data
  ASSERT_EQ(cache.evict(100), 0);
  // release hold blocks
  seqs_data.clear();

  // randomly evict all blocks
  int32_t blocks_left = cache.num_blocks();
  while (blocks_left > 0) {
    // randomly generate number of blocks to evict this round: [1, blocks_left]
    int32_t to_evict = absl::Uniform<int32_t>(gen, 1, blocks_left + 1);
    int32_t evicted = cache.evict(to_evict);
    // evicted should be non-zero, otherwise, it's a deadloop
    ASSERT_GT(evicted, 0);
    // should evicted exactly same number of blocks since no thers hold blocks
    EXPECT_EQ(to_evict, evicted);
    blocks_left -= evicted;
  }

  // all blocks are evicted and return to allocator
  EXPECT_EQ(cache.num_blocks(), 0);
  EXPECT_EQ(allocator.num_free_blocks(), total_blocks);
}

INSTANTIATE_TEST_SUITE_P(
    Random,
    PrefixCacheRandomTest,
    ::testing::Combine(::testing::Values(1, 4, 8, 32, 128, 256),  // block_size
                       ::testing::Values(1000),                   // max_seq_len
                       ::testing::Values(1000)                    // num_seqs
                       ));

}  // namespace llm