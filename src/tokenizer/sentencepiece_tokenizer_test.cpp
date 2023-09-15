#include "sentencepiece_tokenizer.h"

#include <gtest/gtest.h>

namespace llm {

TEST(SentencePieceTokenizerTest, EncodeTest) {
  SentencePieceTokenizer tokenizer("data/test_model.model");
  std::vector<int> ids;
  ASSERT_TRUE(tokenizer.encode("Hello, world!", &ids));
  const std::vector<int> desired_ids = {1, 151, 88, 21, 5, 887, 147};
  EXPECT_EQ(ids, desired_ids);
}

TEST(SentencePieceTokenizerTest, DecodeTest) {
  SentencePieceTokenizer tokenizer("data/test_model.model");
  const std::vector<int> tokens = {1, 151, 88, 21, 5, 887, 147};
  const auto text = tokenizer.decode(tokens);
  EXPECT_EQ(text, "Hello, world!");
}

TEST(SentencePieceTokenizerTest, CJKTest) {
  SentencePieceTokenizer tokenizer("data/test_model.model");
  // const std::wstring test_text = L"你好，世界！";
  // const auto ids = tokenizer.Encode(test_text);
  // const std::vector<int> desired_ids = {4, 0, 5, 0, 147};
  // EXPECT_EQ(ids, desired_ids);

  // const auto text = tokenizer.Decode(ids);
  // EXPECT_EQ(text, test_text);
}
}  // namespace llm
