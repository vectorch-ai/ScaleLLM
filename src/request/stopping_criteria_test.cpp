#include "stopping_criteria.h"

#include <gtest/gtest.h>

namespace llm {

TEST(StoppingCriteriaTest, MaxTokens) {
  const int32_t max_tokens = 100;

  StoppingCriteria stopping_criteria;
  stopping_criteria.max_tokens = max_tokens;
  stopping_criteria.ignore_eos = true;

  std::vector<int32_t> token_ids = {1, 2, 4};
  const size_t num_prompt_tokens = token_ids.size();

  for (int32_t id = 1; id < max_tokens; ++id) {
    token_ids.push_back(id);
    EXPECT_EQ(stopping_criteria.check_finished(token_ids, num_prompt_tokens),
              FinishReason::NONE);
  }
  token_ids.push_back(max_tokens);
  EXPECT_EQ(stopping_criteria.check_finished(token_ids, num_prompt_tokens),
            FinishReason::LENGTH);
}

TEST(StoppingCriteriaTest, EosTokenId) {
  const int32_t eos_token_id = 30;
  std::vector<int32_t> token_ids = {1, 2, 4};
  const size_t num_prompt_tokens = token_ids.size();

  StoppingCriteria stopping_criteria;
  stopping_criteria.max_tokens = 100;
  stopping_criteria.ignore_eos = false;
  stopping_criteria.eos_token_id = eos_token_id;

  for (int32_t id = 1; id < 3; ++id) {
    token_ids.push_back(id);
    EXPECT_EQ(stopping_criteria.check_finished(token_ids, num_prompt_tokens),
              FinishReason::NONE);
  }

  // append eos token
  token_ids.push_back(eos_token_id);
  EXPECT_EQ(stopping_criteria.check_finished(token_ids, num_prompt_tokens),
            FinishReason::STOP);
}

TEST(StoppingCriteriaTest, StopTokenIds) {
  const int32_t stop_token_id = 30;
  std::vector<int32_t> token_ids = {1, 2, 4};
  const size_t num_prompt_tokens = token_ids.size();

  StoppingCriteria stopping_criteria;
  stopping_criteria.max_tokens = 100;
  stopping_criteria.stop_token_ids = {stop_token_id};

  for (int32_t id = 1; id < 3; ++id) {
    token_ids.push_back(id);
    EXPECT_EQ(stopping_criteria.check_finished(token_ids, num_prompt_tokens),
              FinishReason::NONE);
  }

  // append stop token id
  token_ids.push_back(stop_token_id);
  EXPECT_EQ(stopping_criteria.check_finished(token_ids, num_prompt_tokens),
            FinishReason::STOP);
}

TEST(StoppingCriteriaTest, StopSequences) {
  const int32_t stop_token_id = 30;
  std::vector<int32_t> token_ids = {1, 2, 4};
  const size_t num_prompt_tokens = token_ids.size();

  StoppingCriteria stopping_criteria;
  stopping_criteria.max_tokens = 100;
  stopping_criteria.stop_sequences = {{4, 5, 6}, {5, 6, 7}};

  const std::vector<int32_t> token_ids_to_append = {3, 4, 3, 5, 6};

  for (const auto& id : token_ids_to_append) {
    token_ids.push_back(id);
    EXPECT_EQ(stopping_criteria.check_finished(token_ids, num_prompt_tokens),
              FinishReason::NONE);
  }

  token_ids.push_back(7);
  EXPECT_EQ(stopping_criteria.check_finished(token_ids, num_prompt_tokens),
            FinishReason::STOP);
}

}  // namespace llm
