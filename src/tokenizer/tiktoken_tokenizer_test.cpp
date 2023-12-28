#include "tiktoken_tokenizer.h"

#include <gtest/gtest.h>

namespace llm {

TEST(TiktokenTokenizerTest, EncodeDecodeTest) {
  TiktokenTokenizer tokenizer("data/test.tiktoken");
  EXPECT_EQ(tokenizer.vocab_size(), 300);
  const std::string test_text = "Hello world";
  std::vector<int> ids;
  ASSERT_TRUE(tokenizer.encode(test_text, &ids));
  const std::vector<int> desired_ids = {39, 68, 75, 75, 78, 289, 269, 75, 67};
  EXPECT_EQ(ids, desired_ids);

  const auto text = tokenizer.decode(ids);
  EXPECT_EQ(text, test_text);
}

TEST(TiktokenTokenizerTest, CJKTest) {
  TiktokenTokenizer tokenizer("data/test.tiktoken");
  const std::string test_text = "你好 世界";
  std::vector<int> ids;
  ASSERT_TRUE(tokenizer.encode(test_text, &ids));
  const std::vector<int> desired_ids = {
      160, 121, 254, 161, 98, 121, 220, 160, 116, 244, 163, 243, 234};
  EXPECT_EQ(ids, desired_ids);

  const auto decoded_text = tokenizer.decode(ids);
  EXPECT_EQ(decoded_text, test_text);
}

}  // namespace llm
