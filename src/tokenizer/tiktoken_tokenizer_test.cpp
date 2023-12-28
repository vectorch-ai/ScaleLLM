#include "tiktoken_tokenizer.h"

#include <gtest/gtest.h>

namespace llm {

TEST(TiktokenTokenizerTest, EncodeDecodeTest) {
  TiktokenTokenizer tokenizer("data/test.tiktoken");
  EXPECT_EQ(tokenizer.vocab_size(), 300);
  const std::string test_text = "Hello, world!";
  std::vector<int> ids;
  ASSERT_TRUE(tokenizer.encode(test_text, &ids));
  const std::vector<int> desired_ids = {39, 68, 75, 75, 78, 11, 289, 269, 75, 67, 0};
  EXPECT_EQ(ids, desired_ids);

  const auto text = tokenizer.decode(ids);
  EXPECT_EQ(text, test_text);
}

TEST(TiktokenTokenizerTest, CJKTest) {
  TiktokenTokenizer tokenizer("data/test.tiktoken");
  const std::string test_text = "你好，世界！";
  std::vector<int> ids;
  ASSERT_TRUE(tokenizer.encode(test_text, &ids));
  // clang-format off
  const std::vector<int> desired_ids = {
      160, 121, 254, 161, 98, 121, 171, 120, 234, 
      160, 116, 244, 163, 243, 234, 171, 120, 223};
  // clang-format on
  EXPECT_EQ(ids, desired_ids);

  const auto decoded_text = tokenizer.decode(ids);
  EXPECT_EQ(decoded_text, test_text);
}

TEST(TiktokenTokenizerTest, SpecialTokenTest) {
  // clang-format off
  std::vector<std::string> special_tokens = {
    "[gMASK]", "[sMASK]", "sop", "eop",
    "<|system|>", "<|user|>", "<|assistant|>", "<|observation|>"
  };
  // clang-format on
  TiktokenTokenizer tokenizer("data/test.tiktoken", special_tokens);
  EXPECT_EQ(tokenizer.vocab_size(), 300 + special_tokens.size());
  // test encode each special token
  for (const auto& token : special_tokens) {
    std::vector<int> ids;
    ASSERT_TRUE(tokenizer.encode(token, &ids));
    EXPECT_EQ(ids.size(), 1);
    const auto decoded_token = tokenizer.decode(ids);
    EXPECT_EQ(decoded_token, token);
  }

  // test encode text with special tokens
  const std::string test_text =
      "<|system|> Hello world <|user|> Hello <|assistant|>";
  std::vector<int> ids;
  ASSERT_TRUE(tokenizer.encode(test_text, &ids));
  // clang-format off
  const std::vector<int> desired_ids = {
    304, 220, 39, 68, 75,  75,  78, 289, 269, 75, 67, 220, 
    305, 220, 39,  68,  75, 75, 78,  220, 
    306
  };
  // clang-format on
  EXPECT_EQ(ids, desired_ids);

  const auto text = tokenizer.decode(ids);
  EXPECT_EQ(text, test_text);
}

}  // namespace llm
