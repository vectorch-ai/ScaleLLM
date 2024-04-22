#pragma once

#include <cstdint>
#include <string>

#include "common/slice.h"
#include "tokenizer/tokenizer.h"

namespace llm {

// a stateful decoder that can decode tokens incrementally.
class IncrementalDecoder final {
 public:
  IncrementalDecoder(const std::string_view& prompt,
                     size_t num_prompt_tokens,
                     bool echo,
                     bool skip_special_tokens);

  // decode the token ids incrementally
  // return the decoded delta text since last call.
  std::string decode(const Slice<int32_t>& token_ids,
                     const Tokenizer& tokenizer);

  // get the offset of the output text
  size_t output_offset() const { return output_offset_; }

  // get the offset of the prefix text
  size_t prefix_offset() const { return prefix_offset_; }

 private:
  // the original prompt string, used to skip the prompt decoding when streaming
  std::string_view prompt_;

  // the length of the prompt tokens
  size_t num_prompt_tokens_ = 0;

  // whether to skip special tokens when decoding
  bool skip_special_tokens_ = true;

  // variables to keep track of output text, should be accessed by single thread
  // prefix offset is used to defeat cleanup algorithms in the decode which
  // decide to add a space or not based on surrounding tokens.
  size_t prefix_offset_ = 0;
  // all tokens before output_offset_ have been decoded
  size_t output_offset_ = 0;
};

}  // namespace llm
