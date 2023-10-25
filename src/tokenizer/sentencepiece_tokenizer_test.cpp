#include "sentencepiece_tokenizer.h"

#include <gtest/gtest.h>

namespace llm {

TEST(SentencePieceTokenizerTest, EncodeDecodeTest) {
  SentencePieceTokenizer tokenizer("data/tokenizer.model");
  std::vector<int> ids;
  ASSERT_TRUE(tokenizer.encode("Hello, world!", &ids));
  const std::vector<int> desired_ids = {1, 15043, 29892, 3186, 29991};
  EXPECT_EQ(ids, desired_ids);

  const auto text = tokenizer.decode(ids);
  EXPECT_EQ(text, "Hello, world!");
}

TEST(SentencePieceTokenizerTest, CJKTest) {
  SentencePieceTokenizer tokenizer("data/tokenizer.model");
  const std::string test_text = "你好，世界！";
  std::vector<int> ids;
  ASSERT_TRUE(tokenizer.encode(test_text, &ids));
  const std::vector<int> desired_ids = {
      1, 29871, 30919, 31076, 30214, 30793, 30967, 30584};
  EXPECT_EQ(ids, desired_ids);

  const auto decoded_text = tokenizer.decode(ids);
  EXPECT_EQ(decoded_text, test_text);
}
}  // namespace llm
