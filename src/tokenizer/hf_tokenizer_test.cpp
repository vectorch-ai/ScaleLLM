#include "hf_tokenizer.h"

#include <gtest/gtest.h>

namespace llm {

TEST(HFTokenizerTest, EncodeDecodeTest) {
  auto hf_tokenizer = HFTokenizer::from_file("data/tokenizer.json");
  auto& tokenizer = *hf_tokenizer;

  EXPECT_EQ(tokenizer.vocab_size(), 50257);
  const std::string test_text = "Hello, world!";
  std::vector<int> ids;
  ASSERT_TRUE(tokenizer.encode(test_text, &ids));
  const std::vector<int> desired_ids = {15496, 11, 995, 0};
  EXPECT_EQ(ids, desired_ids);

  {
    const auto text = tokenizer.decode(ids, /*skip_special_tokens=*/false);
    EXPECT_EQ(text, test_text);
  }
  {
    const auto text = tokenizer.decode(ids, /*skip_special_tokens=*/true);
    EXPECT_EQ(text, test_text);
  }
}

TEST(HFTokenizerTest, CJKTest) {
  auto hf_tokenizer = HFTokenizer::from_file("data/tokenizer.json");
  auto& tokenizer = *hf_tokenizer;

  const std::string test_text = "你好，世界！";
  std::vector<int> ids;
  ASSERT_TRUE(tokenizer.encode(test_text, &ids));
  // clang-format off
  const std::vector<int> desired_ids = {19526, 254, 25001, 121, 171, 120, 234, 10310, 244, 45911, 234, 171, 120, 223};
  // clang-format on
  EXPECT_EQ(ids, desired_ids);

  {
    const auto decoded_text =
        tokenizer.decode(ids, /*skip_special_tokens=*/false);
    EXPECT_EQ(decoded_text, test_text);
  }
  {
    const auto decoded_text =
        tokenizer.decode(ids, /*skip_special_tokens=*/true);
    EXPECT_EQ(decoded_text, test_text);
  }
}

TEST(HFTokenizerTest, UTF8Test) {
  auto hf_tokenizer = HFTokenizer::from_file("data/tokenizer.json");
  auto& tokenizer = *hf_tokenizer;

  const std::string test_text = "你好";
  std::vector<int> ids;
  ASSERT_TRUE(tokenizer.encode(test_text, &ids));
  const std::vector<int> desired_ids = {19526, 254, 25001, 121};
  EXPECT_EQ(ids, desired_ids);

  {
    const auto decoded_text =
        tokenizer.decode(ids, /*skip_special_tokens=*/false);
    EXPECT_EQ(decoded_text, test_text);
  }

  // test unfinished utf-8
  {
    const std::string unfinished_decoded_text = "你�";
    const std::vector<int> unfinished_ids = {160, 121, 254, 161, 98};
    const auto decoded_text =
        tokenizer.decode(unfinished_ids, /*skip_special_tokens=*/false);
    EXPECT_EQ(decoded_text, unfinished_decoded_text);
  }
}

}  // namespace llm
