#pragma once
#include <absl/strings/str_join.h>

#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include "common/arg.h"

namespace llm {

struct TokenizerArgs {
  // Type of tokenizer to use. valid values are "sentencepiece" and "tiktoken".
  DEFINE_ARG(std::string, tokenizer_type) = "sentencepiece";

  // Vocab file name.
  DEFINE_ARG(std::string, vocab_file) = "tokenizer.model";

  // Special tokens to add to the vocabulary.
  DEFINE_ARG(std::vector<std::string>, special_tokens);

  // Regex pattern used by tiktok tokenizer only.
  DEFINE_ARG(std::string, pattern);

  // Start id for special tokens. If not set, the start id will be set to the
  // vocab size.
  DEFINE_ARG(std::optional<int32_t>, special_start_id);

  // tokens to add to the beginning of the input sequence.
  DEFINE_ARG(std::vector<std::string>, prefix_tokens);
};

inline std::ostream& operator<<(std::ostream& os, const TokenizerArgs& args) {
  os << "TokenizerArgs: [";
  os << "tokenizer_type: " << args.tokenizer_type();
  os << ", vocab_file: " << args.vocab_file();
  if (!args.special_tokens().empty()) {
    os << ", special_tokens: [" << absl::StrJoin(args.special_tokens(), ", ")
       << "]";
  }
  os << ", pattern: " << args.pattern();
  os << ", special_start_id: " << args.special_start_id().value_or(-1);
  if (!args.prefix_tokens().empty()) {
    os << ", prefix_tokens: [" << absl::StrJoin(args.prefix_tokens(), ", ")
       << "]";
  }
  os << "]";
  return os;
}

}  // namespace llm
