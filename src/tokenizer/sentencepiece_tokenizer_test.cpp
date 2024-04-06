#include "sentencepiece_tokenizer.h"

#include <gtest/gtest.h>

namespace llm {

TEST(SentencePieceTokenizerTest, EncodeDecodeTest) {
  TokenizerArgs args;
  args.vocab_file() = "tokenizer.model";
  args.prefix_tokens() = {"<s>"};
  SentencePieceTokenizer tokenizer("data", args);
  EXPECT_EQ(tokenizer.vocab_size(), 32000);
  const std::string test_text = "Hello, world!";
  std::vector<int> ids;
  ASSERT_TRUE(tokenizer.encode(test_text, &ids));
  const std::vector<int> desired_ids = {1, 15043, 29892, 3186, 29991};
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

TEST(SentencePieceTokenizerTest, CJKTest) {
  TokenizerArgs args;
  args.vocab_file() = "tokenizer.model";
  args.prefix_tokens() = {"<s>"};
  SentencePieceTokenizer tokenizer("data", args);
  EXPECT_EQ(tokenizer.vocab_size(), 32000);
  const std::string test_text = "你好，世界！";
  std::vector<int> ids;
  ASSERT_TRUE(tokenizer.encode(test_text, &ids));
  const std::vector<int> desired_ids = {
      1, 29871, 30919, 31076, 30214, 30793, 30967, 30584};
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

TEST(SentencePieceTokenizerTest, SpecialTokenTest) {
  // clang-format off
  std::vector<std::string> special_tokens = {
    "[gMASK]", "[sMASK]", "sop", "eop",
    "<|system|>", "<|user|>", "<|assistant|>", "<|observation|>"
  };
  // clang-format on
  TokenizerArgs args;
  args.vocab_file() = "tokenizer.model";
  args.special_tokens() = special_tokens;
  SentencePieceTokenizer tokenizer("data", args);
  EXPECT_EQ(tokenizer.vocab_size(), 32000 + special_tokens.size());
  // test encode each special token
  for (const auto& token : special_tokens) {
    std::vector<int> ids;
    ASSERT_TRUE(tokenizer.encode(token, &ids));
    EXPECT_EQ(ids.size(), 1);
    {
      const auto decoded_token =
          tokenizer.decode(ids, /*skip_special_tokens=*/false);
      EXPECT_EQ(decoded_token, token);
    }
    {
      const auto decoded_token =
          tokenizer.decode(ids, /*skip_special_tokens=*/true);
      EXPECT_EQ(decoded_token, "");
    }
  }

  // test encode text with special tokens
  const std::string test_text =
      "<|system|> Hello world <|user|> Hello <|assistant|>";
  std::vector<int> ids;
  ASSERT_TRUE(tokenizer.encode(test_text, &ids));
  // clang-format off
  const std::vector<int> desired_ids = {
    32004, 29871, 15043, 3186, 29871, 
    32005, 29871, 15043, 29871, 
    32006
  };
  // clang-format on
  EXPECT_EQ(ids, desired_ids);

  {
    const auto text = tokenizer.decode(ids, /*skip_special_tokens=*/false);
    EXPECT_EQ(text, test_text);
  }
  {
    const auto text = tokenizer.decode(ids, /*skip_special_tokens=*/true);
    EXPECT_EQ(text, " Hello world  Hello ");
  }
}
}  // namespace llm
