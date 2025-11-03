#pragma once

#include <common/slice.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace llm {

// Fundamentally, Large Language Models (LLM) are designed to generate text
// based on given prompts. To process text effectively, LLM models typically
// work with sequences of integers as inputs and produce sequences of integers
// as outputs. The conversion between text and integer sequences is handled by a
// tokenizer during preprocessing. The tokenizer serves two primary functions:
// 1. Breaking down text into tokens and then mapping those tokens to
// corresponding integers using a predefined vocabulary.
// 2. Reversing this process by converting a sequence of integers back into
// human-readable text using the same vocabulary.
//
//
// For example:
//  ids = tokenizer.Encode("Hello, world!") # [1, 2, 3]
//  text = tokenizer.Decode(ids) # "Hello, world!"
class Tokenizer {
 public:
  virtual ~Tokenizer() = default;

  virtual bool encode(const std::string_view& text,
                      std::vector<int32_t>* ids) const = 0;

  virtual std::string decode(const Slice<int32_t>& ids,
                             bool skip_special_tokens) const = 0;

  virtual std::optional<int32_t> token_to_id(
      const std::string_view& token) const = 0;

  virtual std::string id_to_token(int32_t id) const = 0;

  virtual size_t vocab_size() const = 0;

  virtual std::unique_ptr<Tokenizer> clone() const = 0;
};

}  // namespace llm
