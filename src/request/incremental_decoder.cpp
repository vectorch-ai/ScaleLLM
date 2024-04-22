#include "incremental_decoder.h"

#include <absl/strings/match.h>

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>

#include "common/slice.h"
#include "tokenizer/tokenizer.h"

namespace llm {

IncrementalDecoder::IncrementalDecoder(const std::string_view& prompt,
                                       size_t num_prompt_tokens,
                                       bool echo,
                                       bool skip_special_tokens)
    : prompt_(prompt),
      num_prompt_tokens_(num_prompt_tokens),
      skip_special_tokens_(skip_special_tokens) {
  // if echo is true, set prefix_offset_ and output_offset_ to 0 to print the
  // whole sequence, otherwise set them to the length of the prompt to skip the
  // prompt.
  prefix_offset_ = echo ? 0 : num_prompt_tokens_;
  output_offset_ = echo ? 0 : num_prompt_tokens_;
}

std::string IncrementalDecoder::decode(const Slice<int32_t>& token_ids,
                                       const Tokenizer& tokenizer) {
  std::stringstream ss;
  // return prompt directly if prompt string is not empty
  if (output_offset_ < num_prompt_tokens_ && !prompt_.empty()) {
    // leave 6 tokens for the prefix to defeat cleanup algorithms in decode
    // which decide to add a space or not depending on the surrouding ids.
    prefix_offset_ = num_prompt_tokens_ <= 6 ? 0 : num_prompt_tokens_ - 6;
    output_offset_ = num_prompt_tokens_;
    ss << prompt_;
  }

  const auto prefix_text = tokenizer.decode(
      token_ids.slice(prefix_offset_, output_offset_), skip_special_tokens_);
  const auto new_text =
      tokenizer.decode(token_ids.slice(prefix_offset_), skip_special_tokens_);
  // utf-8 char � at the end means it is a potential unfinished byte sequence
  // from byte fallback tokenization.
  if (new_text.size() > prefix_text.size() && !absl::EndsWith(new_text, "�")) {
    prefix_offset_ = output_offset_;
    output_offset_ = token_ids.size();
    // only print the delta text
    ss << new_text.substr(prefix_text.size());
  }
  return ss.str();
}

}  // namespace llm
