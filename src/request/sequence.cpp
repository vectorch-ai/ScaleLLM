#include "sequence.h"

#include <absl/strings/match.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "tokenizer/tokenizer.h"

namespace llm {

std::atomic<int64_t> Sequence::next_id_{1};

Sequence::Sequence(std::string prompt,
                   std::vector<int32_t> token_ids,
                   const SamplingParameter* sampling_param,
                   const StoppingCriteria* stopping_criteria,
                   OnStream on_stream,
                   bool echo)
    : id_(next_id_.fetch_add(1)),
      prompt_(std::move(prompt)),
      token_ids_(std::move(token_ids)),
      num_prompt_tokens_(token_ids_.size()),
      sampling_param_(sampling_param),
      stopping_criteria_(stopping_criteria),
      on_stream_(on_stream) {
  // reserve enough space for the token ids to avoid reallocation
  // so that the token ids are not invalidated
  const size_t max_tokens = stopping_criteria_->max_tokens;
  token_ids_.reserve(max_tokens + token_ids_.size());
  prefix_offset_ = echo ? 0 : token_ids_.size();
  output_offset_ = echo ? 0 : token_ids_.size();
}

bool Sequence::check_stopping_creteria() {
  if (is_finished_) {
    return true;
  }
  // check against stopping criterias
  const size_t generated_tokens = token_ids_.size() - num_prompt_tokens_;
  const size_t max_new_tokens = stopping_criteria_->max_tokens;
  if (max_new_tokens > 0 && generated_tokens >= max_new_tokens) {
    finish_reason_ = FinishReason::LENGTH;
    return is_finished_ = true;
  }

  if (!stopping_criteria_->ignore_eos_token &&
      token_ids_.back() == stopping_criteria_->eos_token_id) {
    finish_reason_ = FinishReason::STOP;
    return is_finished_ = true;
  }
  // TODO: Add other stopping criterias

  return false;
}

// decode the sequence to get delta text using the tokenizer
std::string Sequence::decode_delta_text(size_t end,
                                        const Tokenizer& tokenizer) {
  const auto prefix_text =
      tokenizer.decode(sub_token_ids(prefix_offset_, output_offset_));
  const auto new_text = tokenizer.decode(sub_token_ids(prefix_offset_, end));
  // utf-8 char � at the end means it is a potential unfinished byte sequence
  // from byte fallback tokenization.
  if (new_text.size() > prefix_text.size() && !absl::EndsWith(new_text, "�")) {
    prefix_offset_ = output_offset_;
    output_offset_ = end;
    // only print the delta text
    return new_text.substr(prefix_text.size());
  }
  return "";
}

}  // namespace llm
