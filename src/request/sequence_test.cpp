#include "sequence.h"

#include <absl/time/clock.h>
#include <gtest/gtest.h>

#include "memory/block.h"
#include "request/stopping_criteria.h"
#include "sampling/parameters.h"

namespace llm {

TEST(SequenceTest, DiscardTokenIds) {
  const int32_t max_tokens = 100;
  std::vector<int32_t> prompt_tokens = {1, 2, 4};
  Sequence::Options options;
  options.stopping_criteria.max_tokens = max_tokens;
  Sequence sequence(/*prompt=*/"",
                    prompt_tokens,
                    /*capacity=*/200,
                    options);
  EXPECT_EQ(sequence.num_prompt_tokens(), 3);
  EXPECT_EQ(sequence.num_generated_tokens(), 0);
  EXPECT_EQ(sequence.num_tokens(), 3);

  // append tokens to prefill sequence expect CHECK failure
  EXPECT_DEATH(sequence.append_token(10),
               "cannot append token to a prefill sequence");

  EXPECT_EQ(sequence.num_prompt_tokens(), 3);
  EXPECT_EQ(sequence.num_generated_tokens(), 0);
  EXPECT_EQ(sequence.num_tokens(), 3);
  EXPECT_EQ(sequence.token_ids(), prompt_tokens);

  // allocate block and commit kv_cache
  sequence.append_block({/*id=*/0, /*size=*/20});
  sequence.commit_kv_cache(prompt_tokens.size());

  // all following tokens will be appended
  sequence.append_token(40);
  sequence.append_token(50);

  EXPECT_EQ(sequence.num_prompt_tokens(), 3);
  EXPECT_EQ(sequence.num_generated_tokens(), 2);
  EXPECT_EQ(sequence.num_tokens(), 5);

  std::vector<int32_t> desired_tokens = {1, 2, 4, 40, 50};
  EXPECT_EQ(sequence.token_ids(), desired_tokens);
}

TEST(SequenceTest, Speculative4StepsPartiallyMatch) {
  // test scenarios speculative decoding
  const int32_t max_tokens = 100;
  std::vector<int32_t> prompt_tokens = {1, 2, 4};
  Sequence::Options options;
  options.stopping_criteria.max_tokens = max_tokens;
  Sequence sequence(/*prompt=*/"",
                    prompt_tokens,
                    /*capacity=*/200,
                    options);
  EXPECT_EQ(sequence.num_prompt_tokens(), 3);
  EXPECT_EQ(sequence.num_generated_tokens(), 0);
  EXPECT_EQ(sequence.num_tokens(), 3);

  // allocate block
  sequence.append_block({/*id=*/0, /*size=*/200});

  // kv cache in sync for draft and target
  {
    sequence.set_engine_type(EngineType::SSM);
    sequence.commit_kv_cache(prompt_tokens.size() - 1);

    sequence.set_engine_type(EngineType::LLM);
    sequence.commit_kv_cache(prompt_tokens.size() - 1);
  }
  EXPECT_EQ(sequence.num_kv_cache_tokens(EngineType::SSM), 2);
  EXPECT_EQ(sequence.num_kv_cache_tokens(EngineType::LLM), 2);

  const std::vector<int32_t> spec_token_ids = {1058, 338, 4473, 29973};
  const int32_t bonus_token_id = 4343;
  const uint64_t num_spec_tokens = 4;
  // add draft tokens
  {
    sequence.set_engine_type(EngineType::SSM);
    for (int i = 0; i < num_spec_tokens; i++) {
      // run engine and update kv cache pos
      sequence.commit_kv_cache(sequence.num_tokens_to_process());
      // append new token
      sequence.append_token(spec_token_ids[i]);
    }
    const std::vector<int32_t> desired_tokens = {
        1, 2, 4, 1058, 338, 4473, 29973};
    EXPECT_EQ(sequence.token_ids(), desired_tokens);
    EXPECT_EQ(sequence.num_kv_cache_tokens(), 6);
  }

  // validate with accepted tokens
  {
    sequence.set_engine_type(EngineType::LLM);

    // run engine and update kv cache pos
    sequence.commit_kv_cache(sequence.num_tokens_to_process());

    // append bonus token
    sequence.append_token(bonus_token_id);
    const std::vector<int32_t> desired_tokens = {
        1, 2, 4, 1058, 338, 4473, 29973, bonus_token_id};
    EXPECT_EQ(sequence.token_ids(), desired_tokens);
    EXPECT_EQ(sequence.num_kv_cache_tokens(), 7);

    const std::vector<int64_t> accepted_token_ids = {1058, 338, 4473, 1058, -1};
    const size_t num_accepted_tokens =
        sequence.validate_tokens(accepted_token_ids);
    EXPECT_EQ(num_accepted_tokens, 4);

    const std::vector<int32_t> validated_tokens = {
        1, 2, 4, 1058, 338, 4473, 1058};
    EXPECT_EQ(sequence.token_ids(), validated_tokens);

    // check kv caches
    EXPECT_EQ(sequence.num_kv_cache_tokens(EngineType::SSM), 6);
    EXPECT_EQ(sequence.num_kv_cache_tokens(EngineType::LLM), 6);
  }
}

TEST(SequenceTest, Speculative4StepsFullMatch) {
  // test scenarios speculative decoding
  const int32_t max_tokens = 100;
  std::vector<int32_t> prompt_tokens = {1, 2, 4};
  Sequence::Options options;
  options.stopping_criteria.max_tokens = max_tokens;
  Sequence sequence(/*prompt=*/"",
                    prompt_tokens,
                    /*capacity=*/200,
                    options);
  EXPECT_EQ(sequence.num_prompt_tokens(), 3);
  EXPECT_EQ(sequence.num_generated_tokens(), 0);
  EXPECT_EQ(sequence.num_tokens(), 3);

  // allocate block
  sequence.append_block({/*id=*/0, /*size=*/200});

  // kv cache in sync for draft and target
  {
    sequence.set_engine_type(EngineType::SSM);
    sequence.commit_kv_cache(prompt_tokens.size() - 1);

    sequence.set_engine_type(EngineType::LLM);
    sequence.commit_kv_cache(prompt_tokens.size() - 1);
  }
  EXPECT_EQ(sequence.num_kv_cache_tokens(EngineType::SSM), 2);
  EXPECT_EQ(sequence.num_kv_cache_tokens(EngineType::LLM), 2);

  const std::vector<int32_t> spec_token_ids = {1058, 338, 4473, 29973};
  const int32_t bonus_token_id = 4343;
  const uint64_t num_spec_tokens = 4;
  // add draft tokens
  {
    sequence.set_engine_type(EngineType::SSM);
    for (int i = 0; i < num_spec_tokens; i++) {
      // run engine and update kv cache pos
      sequence.commit_kv_cache(sequence.num_tokens_to_process());
      // append new token
      sequence.append_token(spec_token_ids[i]);
    }
    const std::vector<int32_t> desired_tokens = {
        1, 2, 4, 1058, 338, 4473, 29973};
    EXPECT_EQ(sequence.token_ids(), desired_tokens);
    EXPECT_EQ(sequence.num_kv_cache_tokens(), 6);
  }

  // validate with accepted tokens
  {
    sequence.set_engine_type(EngineType::LLM);

    // run engine and update kv cache pos
    sequence.commit_kv_cache(sequence.num_tokens_to_process());

    // append bonus token
    sequence.append_token(bonus_token_id);
    const std::vector<int32_t> desired_tokens = {
        1, 2, 4, 1058, 338, 4473, 29973, bonus_token_id};
    EXPECT_EQ(sequence.token_ids(), desired_tokens);
    EXPECT_EQ(sequence.num_kv_cache_tokens(), 7);

    const std::vector<int64_t> accepted_token_ids = {
        1058, 338, 4473, 29973, bonus_token_id};
    const size_t num_accepted_tokens =
        sequence.validate_tokens(accepted_token_ids);
    EXPECT_EQ(num_accepted_tokens, 5);

    const std::vector<int32_t> validated_tokens = {
        1, 2, 4, 1058, 338, 4473, 29973, bonus_token_id};
    EXPECT_EQ(sequence.token_ids(), validated_tokens);

    // check kv caches, LLM should have one more token in kv cache
    EXPECT_EQ(sequence.num_kv_cache_tokens(EngineType::SSM), 6);
    EXPECT_EQ(sequence.num_kv_cache_tokens(EngineType::LLM), 7);
  }
}

TEST(SequenceTest, SpeculativeNoMatch) {
  // test scenarios speculative decoding
  const int32_t max_tokens = 100;
  std::vector<int32_t> prompt_tokens = {1, 2, 4};
  Sequence::Options options;
  options.stopping_criteria.max_tokens = max_tokens;
  Sequence sequence(/*prompt=*/"",
                    prompt_tokens,
                    /*capacity=*/200,
                    options);
  EXPECT_EQ(sequence.num_prompt_tokens(), 3);
  EXPECT_EQ(sequence.num_generated_tokens(), 0);
  EXPECT_EQ(sequence.num_tokens(), 3);

  // allocate block
  sequence.append_block({/*id=*/0, /*size=*/200});

  // kv cache in sync for draft and target
  {
    sequence.set_engine_type(EngineType::SSM);
    sequence.commit_kv_cache(prompt_tokens.size() - 1);

    sequence.set_engine_type(EngineType::LLM);
    sequence.commit_kv_cache(prompt_tokens.size() - 1);
  }
  EXPECT_EQ(sequence.num_kv_cache_tokens(EngineType::SSM), 2);
  EXPECT_EQ(sequence.num_kv_cache_tokens(EngineType::LLM), 2);

  const std::vector<int32_t> spec_token_ids = {1058, 338, 4473, 29973};
  const int32_t bonus_token_id = 4343;
  const uint64_t num_spec_tokens = 4;
  // add draft tokens
  {
    sequence.set_engine_type(EngineType::SSM);
    for (int i = 0; i < num_spec_tokens; i++) {
      // run engine and update kv cache pos
      sequence.commit_kv_cache(sequence.num_tokens_to_process());
      // append new token
      sequence.append_token(spec_token_ids[i]);
    }
    const std::vector<int32_t> desired_tokens = {
        1, 2, 4, 1058, 338, 4473, 29973};
    EXPECT_EQ(sequence.token_ids(), desired_tokens);
    EXPECT_EQ(sequence.num_kv_cache_tokens(), 6);
  }

  // validate with accepted tokens
  {
    sequence.set_engine_type(EngineType::LLM);

    // run engine and update kv cache pos
    sequence.commit_kv_cache(sequence.num_tokens_to_process());

    // append bonus token
    sequence.append_token(bonus_token_id);
    const std::vector<int32_t> desired_tokens = {
        1, 2, 4, 1058, 338, 4473, 29973, bonus_token_id};
    EXPECT_EQ(sequence.token_ids(), desired_tokens);
    EXPECT_EQ(sequence.num_kv_cache_tokens(), 7);

    const std::vector<int64_t> accepted_token_ids = {1058, -1, -1, -1, -1};
    const size_t num_accepted_tokens =
        sequence.validate_tokens(accepted_token_ids);
    EXPECT_EQ(num_accepted_tokens, 1);

    const std::vector<int32_t> validated_tokens = {1, 2, 4, 1058};
    EXPECT_EQ(sequence.token_ids(), validated_tokens);

    // check kv caches, LLM should have one more token in kv cache
    EXPECT_EQ(sequence.num_kv_cache_tokens(EngineType::SSM), 3);
    EXPECT_EQ(sequence.num_kv_cache_tokens(EngineType::LLM), 3);
  }
}

}  // namespace llm
