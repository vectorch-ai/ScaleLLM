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
  EXPECT_DEATH(sequence.append_new_token_id(10),
               "cannot append token to a prefill sequence");

  EXPECT_EQ(sequence.num_prompt_tokens(), 3);
  EXPECT_EQ(sequence.num_generated_tokens(), 0);
  EXPECT_EQ(sequence.num_tokens(), 3);
  EXPECT_EQ(sequence.token_ids(), prompt_tokens);

  // allocate block
  sequence.append_block({/*id=*/0, /*size=*/20});
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
