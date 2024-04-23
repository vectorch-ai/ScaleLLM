#include "sequence.h"

#include <absl/time/clock.h>
#include <gtest/gtest.h>

#include "memory/block.h"
#include "request/stopping_criteria.h"
#include "sampling/parameters.h"

namespace llm {
namespace {
void run_speculative_decoding(Sequence& sequence,
                              const std::vector<int32_t>& draft_token_ids,
                              const int32_t bonus_token_id,
                              const int32_t resample_token_id,
                              const size_t num_accepted_tokens) {
  const size_t num_spec_tokens = draft_token_ids.size();
  EXPECT_LE(num_accepted_tokens, num_spec_tokens);

  // kv cache in sync for draft and target
  const size_t ssm_kv_cache_size =
      sequence.num_kv_cache_tokens(EngineType::SSM);
  const size_t llm_kv_cache_size =
      sequence.num_kv_cache_tokens(EngineType::LLM);
  EXPECT_GE(llm_kv_cache_size, ssm_kv_cache_size);
  const size_t kv_diff = llm_kv_cache_size - ssm_kv_cache_size;
  // at most one token difference between LLM and SSM for speculative decoding
  EXPECT_LE(kv_diff, 1);

  // remember the initial tokens
  const auto initial_tokens = sequence.token_ids();
  // build desired tokens
  std::vector<int32_t> desired_tokens = initial_tokens;
  // add draft_token_ids into desired_tokens
  desired_tokens.insert(
      desired_tokens.end(), draft_token_ids.begin(), draft_token_ids.end());

  // build accepted tokens
  std::vector<int64_t> accepted_token_ids = {draft_token_ids.begin(),
                                             draft_token_ids.end()};
  accepted_token_ids.push_back(bonus_token_id);
  // add resample token if needed
  if (num_accepted_tokens < draft_token_ids.size()) {
    // replace the last token with resample token
    accepted_token_ids[num_accepted_tokens] = resample_token_id;
  }
  // mask out remaining tokens
  for (size_t i = num_accepted_tokens + 1; i < accepted_token_ids.size(); i++) {
    accepted_token_ids[i] = -1;
  }

  size_t draft_processed_tokens = 0;
  size_t target_processed_tokens = 0;
  // add draft tokens
  {
    sequence.set_engine_type(EngineType::SSM);
    for (size_t i = 0; i < num_spec_tokens; i++) {
      const size_t num_tokens_to_process = sequence.num_tokens_to_process();
      draft_processed_tokens += num_tokens_to_process;

      // run engine and update kv cache pos
      sequence.commit_kv_cache(num_tokens_to_process);
      // append new token
      sequence.append_token(draft_token_ids[i]);
    }
    EXPECT_EQ(sequence.token_ids(), desired_tokens);
    // only last token is not in kv cache
    EXPECT_EQ(sequence.num_kv_cache_tokens(), desired_tokens.size() - 1);
  }

  // validated with accepted tokens
  {
    sequence.set_engine_type(EngineType::LLM);

    const size_t num_tokens_to_process = sequence.num_tokens_to_process();
    target_processed_tokens += num_tokens_to_process;

    // run engine and update kv cache pos
    sequence.commit_kv_cache(num_tokens_to_process);

    // append bonus token
    sequence.append_token(bonus_token_id);
    desired_tokens.push_back(bonus_token_id);

    EXPECT_EQ(sequence.token_ids(), desired_tokens);
    // only last token is not in kv cache
    EXPECT_EQ(sequence.num_kv_cache_tokens(), desired_tokens.size() - 1);

    // varify with accepted tokens
    EXPECT_EQ(sequence.validate_tokens(accepted_token_ids),
              num_accepted_tokens + 1);

    std::vector<int32_t> validated_tokens = initial_tokens;
    for (size_t i = 0; i < num_accepted_tokens + 1; ++i) {
      EXPECT_NE(accepted_token_ids[i], -1);
      validated_tokens.push_back(accepted_token_ids[i]);
    }

    EXPECT_EQ(sequence.token_ids(), validated_tokens);
  }

  // check processed tokens
  EXPECT_EQ(draft_processed_tokens + 1, target_processed_tokens + kv_diff);

  const auto n_tokens = sequence.num_tokens();
  // check kv caches pos
  if (num_accepted_tokens == num_spec_tokens) {
    // All match: LLM should have one more token in kv cache
    EXPECT_EQ(sequence.num_kv_cache_tokens(EngineType::SSM), n_tokens - 2);
    EXPECT_EQ(sequence.num_kv_cache_tokens(EngineType::LLM), n_tokens - 1);
  } else {
    // LLM should have the same number of tokens in kv cache
    EXPECT_EQ(sequence.num_kv_cache_tokens(EngineType::SSM), n_tokens - 1);
    EXPECT_EQ(sequence.num_kv_cache_tokens(EngineType::LLM), n_tokens - 1);
  }
}
}  // namespace

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

TEST(SequenceTest, SpeculativeBasic) {
  // test scenarios speculative decoding
  std::vector<int32_t> prompt_tokens = {1, 2, 4};
  Sequence::Options options;
  options.stopping_criteria.max_tokens = 100;
  Sequence sequence(/*prompt=*/"",
                    prompt_tokens,
                    /*capacity=*/200,
                    options);

  // allocate block
  sequence.append_block({/*id=*/0, /*size=*/200});

  // run speculative decoding with draft tokens: {1058, 338, 4473, 29973}
  // expect to accept 3 tokens and resample 1 token: {1058, 338, 4473, 1058}
  run_speculative_decoding(sequence,
                           /*draft_token_ids=*/{1058, 338, 4473, 29973},
                           /*bonus_token_id=*/4343,
                           /*resample_token_id=*/1058,
                           /*num_accepted_tokens=*/3);

  const std::vector<int32_t> desired_tokens = {1, 2, 4, 1058, 338, 4473, 1058};
  EXPECT_EQ(sequence.token_ids(), desired_tokens);

  // check kv caches
  EXPECT_EQ(sequence.num_kv_cache_tokens(EngineType::SSM), 6);
  EXPECT_EQ(sequence.num_kv_cache_tokens(EngineType::LLM), 6);
}

TEST(SequenceTest, SpeculativeFullMatch) {
  // test scenarios speculative decoding
  std::vector<int32_t> prompt_tokens = {1, 2, 4};
  Sequence::Options options;
  options.stopping_criteria.max_tokens = 100;
  Sequence sequence(/*prompt=*/"",
                    prompt_tokens,
                    /*capacity=*/200,
                    options);

  // allocate block
  sequence.append_block({/*id=*/0, /*size=*/200});

  for (size_t i = 0; i < 4; ++i) {
    // run speculative decoding with draft tokens: {1058, 338, 4473, 29973}
    // expect to accept all tokens: {1058, 338, 4473, 29973}
    run_speculative_decoding(sequence,
                             /*draft_token_ids=*/{1058, 338, 4473, 29973},
                             /*bonus_token_id=*/4343,
                             /*resample_token_id=*/1058,
                             /*num_accepted_tokens=*/4);
  }

  // clang-format off
  const std::vector<int32_t> validated_tokens = {
      /*prompt*/ 1, 2, 4, 
      /*first*/  1058, 338, 4473, 29973, 4343, 
      /*second*/ 1058, 338, 4473, 29973, 4343, 
      /*third*/  1058, 338, 4473, 29973, 4343, 
      /*forth*/  1058, 338, 4473, 29973, 4343};
  // clang-format on
  EXPECT_EQ(sequence.token_ids(), validated_tokens);

  // check kv caches, LLM should have one more token in kv cache
  EXPECT_EQ(sequence.num_kv_cache_tokens(EngineType::SSM),
            validated_tokens.size() - 2);
  EXPECT_EQ(sequence.num_kv_cache_tokens(EngineType::LLM),
            validated_tokens.size() - 1);
}

TEST(SequenceTest, SpeculativeNoMatch) {
  // test scenarios speculative decoding
  std::vector<int32_t> prompt_tokens = {1, 2, 4};
  Sequence::Options options;
  options.stopping_criteria.max_tokens = 100;
  Sequence sequence(/*prompt=*/"",
                    prompt_tokens,
                    /*capacity=*/200,
                    options);

  // allocate block
  sequence.append_block({/*id=*/0, /*size=*/200});

  for (size_t i = 0; i < 4; ++i) {
    // run speculative decoding with draft tokens: {1058, 338, 4473, 29973}
    // expect to accept no tokens: {1314}
    run_speculative_decoding(sequence,
                             /*draft_token_ids=*/{1058, 338, 4473, 29973},
                             /*bonus_token_id=*/4343,
                             /*resample_token_id=*/1314,
                             /*num_accepted_tokens=*/0);
  }

  // clang-format off
  const std::vector<int32_t> validated_tokens = {
      /*prompt*/ 1, 2, 4, 
      /*first*/  1314, 
      /*second*/ 1314, 
      /*third*/  1314, 
      /*forth*/  1314};
  // clang-format on
  EXPECT_EQ(sequence.token_ids(), validated_tokens);

  // check kv caches, LLM should have one more token in kv cache
  EXPECT_EQ(sequence.num_kv_cache_tokens(EngineType::SSM),
            validated_tokens.size() - 1);
  EXPECT_EQ(sequence.num_kv_cache_tokens(EngineType::LLM),
            validated_tokens.size() - 1);
}

TEST(SequenceTest, SpeculativePartiallyMatch) {
  // test scenarios speculative decoding
  std::vector<int32_t> prompt_tokens = {1, 2, 4};
  Sequence::Options options;
  options.stopping_criteria.max_tokens = 100;
  Sequence sequence(/*prompt=*/"",
                    prompt_tokens,
                    /*capacity=*/200,
                    options);

  // allocate block
  sequence.append_block({/*id=*/0, /*size=*/200});

  for (size_t i = 0; i <= 4; ++i) {
    run_speculative_decoding(sequence,
                             /*draft_token_ids=*/{1058, 338, 4473, 29973},
                             /*bonus_token_id=*/4343,
                             /*resample_token_id=*/1314,
                             /*num_accepted_tokens=*/i);
  }

  // clang-format off
  const std::vector<int32_t> validated_tokens = {
      /*prompt*/ 1, 2, 4, 
      /*first*/  1314, 
      /*second*/ 1058, 1314, 
      /*third*/  1058, 338, 1314, 
      /*forth*/  1058, 338, 4473, 1314,
      /*fifth*/  1058, 338, 4473, 29973, 4343
  };
  // clang-format on
  EXPECT_EQ(sequence.token_ids(), validated_tokens);

  // check kv caches, LLM should have one more token in kv cache
  EXPECT_EQ(sequence.num_kv_cache_tokens(EngineType::SSM),
            validated_tokens.size() - 2);
  EXPECT_EQ(sequence.num_kv_cache_tokens(EngineType::LLM),
            validated_tokens.size() - 1);
}

}  // namespace llm
