#include "sequence.h"

#include <absl/time/clock.h>
#include <gtest/gtest.h>

#include "memory/block.h"
#include "memory/block_allocator.h"
#include "request/stopping_criteria.h"
#include "sampling/parameters.h"

namespace llm {

TEST(SequenceTest, MaxTokens) {
  const uint32_t total_blocks = 20;
  const uint32_t block_size = 20;
  BlockAllocator allocator(total_blocks, block_size);

  const int32_t max_tokens = 100;
  std::vector<int32_t> prompt_tokens = {1, 2, 4};
  StoppingCriteria stopping_criteria;
  stopping_criteria.max_tokens = max_tokens;
  stopping_criteria.ignore_eos_token = true;
  SamplingParameter sampling_param;

  Sequence sequence(prompt_tokens,
                    sampling_param,
                    stopping_criteria,
                    /*echo=*/false,
                    /*on_stream=*/nullptr);
  EXPECT_EQ(sequence.num_prompt_tokens(), 3);
  EXPECT_EQ(sequence.num_generated_tokens(), 0);
  EXPECT_EQ(sequence.num_tokens(), 3);
  sequence.append_blocks(allocator.allocate(10));
  sequence.commit_kv_cache(/*size=*/3);

  for (int32_t id = 1; id < max_tokens; ++id) {
    sequence.append_new_token_id(id);
  }
  sequence.append_new_token_id(max_tokens);
  EXPECT_TRUE(sequence.is_finished());

  EXPECT_EQ(sequence.finish_reason(), FinishReason::LENGTH);

  EXPECT_EQ(sequence.num_prompt_tokens(), 3);
  EXPECT_EQ(sequence.num_generated_tokens(), max_tokens);
  EXPECT_EQ(sequence.num_tokens(), max_tokens + 3);
}

TEST(SequenceTest, EosTokenId) {
  const uint32_t total_blocks = 20;
  const uint32_t block_size = 20;
  BlockAllocator allocator(total_blocks, block_size);

  const int32_t eos_token_id = 30;
  std::vector<int32_t> prompt_tokens = {1, 2, 4};

  StoppingCriteria stopping_criteria;
  stopping_criteria.max_tokens = 10;
  stopping_criteria.ignore_eos_token = false;
  stopping_criteria.eos_token_id = eos_token_id;
  SamplingParameter sampling_param;

  Sequence sequence(prompt_tokens,
                    sampling_param,
                    stopping_criteria,
                    /*echo=*/false,
                    /*on_stream=*/nullptr);
  sequence.append_blocks(allocator.allocate(10));
  sequence.commit_kv_cache(/*size=*/3);

  sequence.append_new_token_id(eos_token_id);
  EXPECT_TRUE(sequence.is_finished());
  EXPECT_EQ(sequence.finish_reason(), FinishReason::STOP);

  EXPECT_EQ(sequence.num_prompt_tokens(), 3);
  EXPECT_EQ(sequence.num_generated_tokens(), 1);
  EXPECT_EQ(sequence.num_tokens(), 4);

  std::vector<int32_t> desired_tokens = {1, 2, 4, eos_token_id};
  EXPECT_EQ(sequence.token_ids(), desired_tokens);
}

TEST(SequenceTest, StopTokenIds) {
  const uint32_t total_blocks = 20;
  const uint32_t block_size = 20;
  BlockAllocator allocator(total_blocks, block_size);
  std::vector<int32_t> prompt_tokens = {1, 2, 4};

  StoppingCriteria stopping_criteria;
  stopping_criteria.max_tokens = 100;
  stopping_criteria.stop_token_ids = {20};

  SamplingParameter sampling_param;

  Sequence sequence(prompt_tokens,
                    sampling_param,
                    stopping_criteria,
                    /*echo=*/false,
                    /*on_stream=*/nullptr);

  EXPECT_EQ(sequence.num_prompt_tokens(), 3);
  EXPECT_EQ(sequence.num_generated_tokens(), 0);
  EXPECT_EQ(sequence.num_tokens(), 3);

  sequence.append_blocks(allocator.allocate(10));
  sequence.commit_kv_cache(/*size=*/3);

  sequence.append_new_token_id(3);
  sequence.append_new_token_id(4);
  sequence.append_new_token_id(3);
  sequence.append_new_token_id(5);
  sequence.append_new_token_id(6);
  sequence.append_new_token_id(20);

  EXPECT_TRUE(sequence.is_finished());
  EXPECT_EQ(sequence.finish_reason(), FinishReason::STOP);

  EXPECT_EQ(sequence.num_prompt_tokens(), 3);
  EXPECT_EQ(sequence.num_generated_tokens(), 6);
  EXPECT_EQ(sequence.num_tokens(), 9);
  std::vector<int32_t> desired_tokens = {1, 2, 4, 3, 4, 3, 5, 6, 20};
  EXPECT_EQ(sequence.token_ids(), desired_tokens);
}

TEST(SequenceTest, StopSequences) {
  const uint32_t total_blocks = 20;
  const uint32_t block_size = 20;
  BlockAllocator allocator(total_blocks, block_size);
  std::vector<int32_t> prompt_tokens = {1, 2, 4};

  StoppingCriteria stopping_criteria;
  stopping_criteria.max_tokens = 100;
  stopping_criteria.stop_sequences = {{4, 5, 6}, {5, 6, 7}};

  SamplingParameter sampling_param;

  Sequence sequence(prompt_tokens,
                    sampling_param,
                    stopping_criteria,
                    /*echo=*/false,
                    /*on_stream=*/nullptr);

  EXPECT_EQ(sequence.num_prompt_tokens(), 3);
  EXPECT_EQ(sequence.num_generated_tokens(), 0);
  EXPECT_EQ(sequence.num_tokens(), 3);

  sequence.append_blocks(allocator.allocate(10));
  sequence.commit_kv_cache(/*size=*/3);

  sequence.append_new_token_id(3);
  sequence.append_new_token_id(4);
  sequence.append_new_token_id(3);
  sequence.append_new_token_id(5);
  sequence.append_new_token_id(6);
  sequence.append_new_token_id(7);

  EXPECT_TRUE(sequence.is_finished());
  EXPECT_EQ(sequence.finish_reason(), FinishReason::STOP);

  EXPECT_EQ(sequence.num_prompt_tokens(), 3);
  EXPECT_EQ(sequence.num_generated_tokens(), 6);
  EXPECT_EQ(sequence.num_tokens(), 9);
  std::vector<int32_t> desired_tokens = {1, 2, 4, 3, 4, 3, 5, 6, 7};
  EXPECT_EQ(sequence.token_ids(), desired_tokens);
}

TEST(SequenceTest, DiscardTokenIds) {
  const uint32_t total_blocks = 20;
  const uint32_t block_size = 20;
  BlockAllocator allocator(total_blocks, block_size);

  const int32_t max_tokens = 100;
  std::vector<int32_t> prompt_tokens = {1, 2, 4};
  StoppingCriteria stopping_criteria;
  stopping_criteria.max_tokens = max_tokens;
  SamplingParameter sampling_param;

  Sequence sequence(prompt_tokens,
                    sampling_param,
                    stopping_criteria,
                    /*echo=*/false,
                    /*on_stream=*/nullptr);
  EXPECT_EQ(sequence.num_prompt_tokens(), 3);
  EXPECT_EQ(sequence.num_generated_tokens(), 0);
  EXPECT_EQ(sequence.num_tokens(), 3);

  // append tokens to prefill sequence expect CHECK failure
  EXPECT_DEATH(sequence.append_new_token_id(10),
               "cannot append token to a prefill sequence");
  EXPECT_DEATH(sequence.append_new_token_id(20),
               "cannot append token to a prefill sequence");
  EXPECT_DEATH(sequence.append_new_token_id(30),
               "cannot append token to a prefill sequence");

  EXPECT_EQ(sequence.num_prompt_tokens(), 3);
  EXPECT_EQ(sequence.num_generated_tokens(), 0);
  EXPECT_EQ(sequence.num_tokens(), 3);
  EXPECT_EQ(sequence.token_ids(), prompt_tokens);

  sequence.append_blocks(allocator.allocate(10));
  sequence.commit_kv_cache(prompt_tokens.size());

  // all following tokens will be appended
  sequence.append_new_token_id(40);
  sequence.append_new_token_id(50);

  EXPECT_EQ(sequence.num_prompt_tokens(), 3);
  EXPECT_EQ(sequence.num_generated_tokens(), 2);
  EXPECT_EQ(sequence.num_tokens(), 5);

  std::vector<int32_t> desired_tokens = {1, 2, 4, 40, 50};
  EXPECT_EQ(sequence.token_ids(), desired_tokens);
}

}  // namespace llm
