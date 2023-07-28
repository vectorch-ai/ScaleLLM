#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace llm {

// Fundamentally, LLM generates text given a prompt. LLM models typically
// take a sequence of integers as input and produces a sequence of integers as
// output. A tokenizer is used in preprocessing steps to convert text to a
// sequence of integers and back. In general, a tokenizer has two main
// functions:
// * breaks text into tokens then maps them to integers using a vocabulary
// * converts a sequence of integers back to text using the vocabulary
// For example:
//  ids = tokenizer.Encode("Hello, world!") # [1, 2, 3]
//  text = tokenizer.Decode(ids) # "Hello, world!"
class Tokenizer {
 public:
  virtual ~Tokenizer() = default;

  virtual std::vector<int> encode(const std::string_view& text) const = 0;

  virtual std::string decode(const std::vector<int>& tokens) const = 0;
};

}  // namespace llm
