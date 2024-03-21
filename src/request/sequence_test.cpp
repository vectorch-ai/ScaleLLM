#include "sequence.h"

#include <absl/time/clock.h>
#include <gtest/gtest.h>

#include "request.h"
#include "request/stopping_criteria.h"
#include "sampling/parameters.h"

namespace llm {

TEST(SequenceTest, MaxTokens) {
  const int32_t max_tokens = 100;
  std::vector<int32_t> prompt_tokens = {1, 2, 4};
  StoppingCriteria stopping_criteria;
  stopping_criteria.max_tokens = max_tokens;
  stopping_criteria.ignore_eos_token = true;
  SamplingParameter sampling_param;

  Sequence sequence(sampling_param,
                    stopping_criteria,
                    prompt_tokens,
                    /*echo=*/false,
                    /*on_stream=*/nullptr);

  EXPECT_EQ(sequence.num_prompt_tokens(), 3);
  EXPECT_EQ(sequence.num_generated_tokens(), 0);
  EXPECT_EQ(sequence.num_tokens(), 3);

  for (int32_t id = 1; id < max_tokens; ++id) {
    EXPECT_TRUE(sequence.append_new_token_id(id));
  }
  EXPECT_FALSE(sequence.append_new_token_id(max_tokens));

  EXPECT_TRUE(sequence.is_finished());
  EXPECT_EQ(sequence.finish_reason(), FinishReason::LENGTH);

  EXPECT_EQ(sequence.num_prompt_tokens(), 3);
  EXPECT_EQ(sequence.num_generated_tokens(), max_tokens);
  EXPECT_EQ(sequence.num_tokens(), max_tokens + 3);
}

TEST(SequenceTest, EosTokenId) {
  const int32_t eos_token_id = 30;
  std::vector<int32_t> prompt_tokens = {1, 2, 4};

  StoppingCriteria stopping_criteria;
  stopping_criteria.max_tokens = 10;
  stopping_criteria.ignore_eos_token = false;
  stopping_criteria.eos_token_id = eos_token_id;
  SamplingParameter sampling_param;

  Sequence sequence(sampling_param,
                    stopping_criteria,
                    prompt_tokens,
                    /*echo=*/false,
                    /*on_stream=*/nullptr);

  EXPECT_FALSE(sequence.append_new_token_id(eos_token_id));

  EXPECT_TRUE(sequence.is_finished());
  EXPECT_EQ(sequence.finish_reason(), FinishReason::STOP);

  EXPECT_EQ(sequence.num_prompt_tokens(), 3);
  EXPECT_EQ(sequence.num_generated_tokens(), 0);
  EXPECT_EQ(sequence.num_tokens(), 3);
}

TEST(SequenceTest, StopTokenIds) {
  std::vector<int32_t> prompt_tokens = {1, 2, 4};

  StoppingCriteria stopping_criteria;
  stopping_criteria.max_tokens = 100;
  stopping_criteria.stop_token_ids = {20};

  SamplingParameter sampling_param;

  Sequence sequence(sampling_param,
                    stopping_criteria,
                    prompt_tokens,
                    /*echo=*/false,
                    /*on_stream=*/nullptr);

  EXPECT_EQ(sequence.num_prompt_tokens(), 3);
  EXPECT_EQ(sequence.num_generated_tokens(), 0);
  EXPECT_EQ(sequence.num_tokens(), 3);

  EXPECT_TRUE(sequence.append_new_token_id(3));
  EXPECT_TRUE(sequence.append_new_token_id(4));
  EXPECT_TRUE(sequence.append_new_token_id(3));
  EXPECT_TRUE(sequence.append_new_token_id(5));
  EXPECT_TRUE(sequence.append_new_token_id(6));
  EXPECT_FALSE(sequence.append_new_token_id(20));

  EXPECT_TRUE(sequence.is_finished());
  EXPECT_EQ(sequence.finish_reason(), FinishReason::STOP);

  EXPECT_EQ(sequence.num_prompt_tokens(), 3);
  EXPECT_EQ(sequence.num_generated_tokens(), 5);
  EXPECT_EQ(sequence.num_tokens(), 8);
}

TEST(SequenceTest, StopSequences) {
  std::vector<int32_t> prompt_tokens = {1, 2, 4};

  StoppingCriteria stopping_criteria;
  stopping_criteria.max_tokens = 100;
  stopping_criteria.stop_sequences = {{4, 5, 6}, {5, 6, 7}};

  SamplingParameter sampling_param;

  Sequence sequence(sampling_param,
                    stopping_criteria,
                    prompt_tokens,
                    /*echo=*/false,
                    /*on_stream=*/nullptr);

  EXPECT_EQ(sequence.num_prompt_tokens(), 3);
  EXPECT_EQ(sequence.num_generated_tokens(), 0);
  EXPECT_EQ(sequence.num_tokens(), 3);

  EXPECT_TRUE(sequence.append_new_token_id(3));
  EXPECT_TRUE(sequence.append_new_token_id(4));
  EXPECT_TRUE(sequence.append_new_token_id(3));
  EXPECT_TRUE(sequence.append_new_token_id(5));
  EXPECT_TRUE(sequence.append_new_token_id(6));
  EXPECT_FALSE(sequence.append_new_token_id(7));

  EXPECT_TRUE(sequence.is_finished());
  EXPECT_EQ(sequence.finish_reason(), FinishReason::STOP);

  EXPECT_EQ(sequence.num_prompt_tokens(), 3);
  EXPECT_EQ(sequence.num_generated_tokens(), 6);
  EXPECT_EQ(sequence.num_tokens(), 9);
}

}  // namespace llm
