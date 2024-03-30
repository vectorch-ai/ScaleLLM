#include "batch.h"

#include <common/logging.h>
#include <gtest/gtest.h>

#include <cstdint>

#include "memory/block.h"
#include "memory/block_allocator.h"
#include "request/stopping_criteria.h"
#include "sampling/parameters.h"

namespace llm {

template <typename T>
bool equal(const torch::Tensor& t, const std::vector<T>& d) {
  auto flatten_t = t.flatten();
  if (flatten_t.size(0) != d.size()) {
    return false;
  }
  for (int i = 0; i < d.size(); i++) {
    if (flatten_t[i].item<T>() != d[i]) {
      return false;
    }
  }
  return true;
}

TEST(BatchTest, Basic) {
  const uint32_t n_blocks = 20;
  const uint32_t block_size = 4;

  BlockAllocator allocator(n_blocks, block_size);
  // reserve block 0
  auto block_0 = allocator.allocate();
  EXPECT_EQ(block_0.id(), 0);

  SamplingParameter sampling_param;
  StoppingCriteria stopping_criteria;
  stopping_criteria.max_tokens = 20;

  // prepare sequences
  // sequence in prefill phase
  Sequence seq1(/*token_ids=*/{1, 3, 5, 7, 5, 4, 3, 2, 1},
                sampling_param,
                stopping_criteria,
                /*echo=*/false,
                /*on_stream=*/nullptr);
  seq1.append_blocks(allocator.allocate(3));  // [1, 2, 3]

  // seq in decode phase
  Sequence seq2(/*token_ids=*/{2, 4, 6, 8, 6, 4, 2},
                sampling_param,
                stopping_criteria,
                /*echo=*/false,
                /*on_stream=*/nullptr);
  seq2.append_blocks(allocator.allocate(4));  // [4, 5, 6, 7]
  seq2.commit_kv_cache(/*size=*/7);
  seq2.append_new_token_id(100);

  // seq in decode phase
  Sequence seq3(
      /*token_ids=*/{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19},
      sampling_param,
      stopping_criteria,
      /*echo=*/false,
      /*on_stream=*/nullptr);
  seq3.append_blocks(allocator.allocate(5));  // [8, 9, 10, 11, 12]
  seq3.commit_kv_cache(/*size=*/15);
  seq3.append_new_token_id(200);

  // define outputs
  Batch batch({&seq1, &seq2, &seq3});
  ModelInput model_input = batch.prepare_model_input();

  // check kv cache pos in sequence
  EXPECT_EQ(seq1.kv_cache_size(), 9);
  EXPECT_EQ(seq2.kv_cache_size(), 8);
  EXPECT_EQ(seq3.kv_cache_size(), 16);

  // clang-format off
  // check the flatten token ids
  const std::vector<int32_t> expcted_tokens = {
      /*seq1*/ 1, 3, 5, 7, 5, 4, 3, 2, 1, 
      /*seq2*/ 100, 
      /*seq3*/ 200};
  EXPECT_TRUE(equal(model_input.token_ids, expcted_tokens));

  // check the flatten positions
  const std::vector<int32_t> expected_pos = {
    /*seq1*/ 0, 1, 2, 3, 4, 5, 6, 7, 8,
    /*seq2*/ 7, 
    /*seq3*/ 15};
  EXPECT_TRUE(equal(model_input.positions, expected_pos));

  // check the input parameters
  const InputParameters& input_params = model_input.input_params;
  EXPECT_FALSE(input_params.empty_kv_cache);
  EXPECT_EQ(input_params.num_sequences, 3);
  EXPECT_EQ(input_params.q_max_seq_len, 9);
  EXPECT_EQ(input_params.kv_max_seq_len, 16);

  const std::vector<int32_t> q_cu_seq_lens = {0, 9, 10, 11};
  EXPECT_TRUE(equal(input_params.q_cu_seq_lens, q_cu_seq_lens));

  const std::vector<int32_t> kv_cu_seq_lens = {0, 9, 17, 33};
  EXPECT_TRUE(equal(input_params.kv_cu_seq_lens, kv_cu_seq_lens));

  const std::vector<int32_t> new_cache_slots = {
    /*seq1*/ 4, 5, 6, 7, 8, 9, 10, 11, 12, 
    /*seq2*/ 23, 
    /*seq3*/ 47};
  EXPECT_TRUE(equal(input_params.new_cache_slots, new_cache_slots));

  const std::vector<int32_t> block_tables = {
    /*seq1*/ 1, 2, 3,  0,  0,
    /*seq2*/ 4, 5, 6,  7,  0,
    /*seq3*/ 8, 9, 10, 11, 12};
  EXPECT_TRUE(equal(input_params.block_tables, block_tables));

  // const std::vector<int32_t> last_token_idxes = {8, 9, 10};
  // EXPECT_TRUE(equal(input_params.last_token_idxes, last_token_idxes));

  const auto& sampling_params = model_input.sampling_params;
  const std::vector<int64_t> unique_ids = {
    /*seq1*/   2,  4,  7,  5,  3,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    /*seq2*/ 100,  8,  6,  4,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 
    /*seq3*/ 200,  19, 17, 1,  2,  15, 3,  4,  5,  6,  7,  8,  9, 10, 11, 13
    };
  EXPECT_TRUE(equal(sampling_params.token_ids, unique_ids));

  const std::vector<int32_t> unique_counts = {
    /*seq1*/  1,  1,  1,  2,  2,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  
    /*seq2*/  1,  1,  2,  2,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  
    /*seq3*/  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1
  };
  EXPECT_TRUE(equal(sampling_params.token_counts, unique_counts));

  const std::vector<int32_t> token_ids_lens = {6, 5, 16};
  EXPECT_TRUE(equal(sampling_params.token_ids_lens, token_ids_lens));

  // clang-format on
}

}  // namespace llm
