#include "hf_tokenizer.h"

#include <gtest/gtest.h>

namespace llm {

TEST(HFTokenizerTest, EncodeTest) {
  auto tokenizer = HFTokenizer::from_file("data/tokenizer.json");
  std::vector<int32_t> ids;
  ASSERT_TRUE(tokenizer->encode("Hello, world!", &ids));
  const std::vector<int32_t> desired_ids = {1, 15043, 29892, 3186, 29991};
  EXPECT_EQ(ids, desired_ids);
}

TEST(HFTokenizerTest, DecodeTest) {
  auto tokenizer = HFTokenizer::from_file("data/tokenizer.json");
  const std::vector<int32_t> tokens = {1, 15043, 29892, 3186, 29991};
  const auto text = tokenizer->decode(tokens);
  EXPECT_EQ(text, "Hello, world!");
}

TEST(HFTokenizerTest, VocabSizeTest) {
  auto tokenizer = HFTokenizer::from_file("data/tokenizer.json");
  EXPECT_EQ(tokenizer->vocab_size(), 32016);
}

}  // namespace llm
