#include "sequence.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "tokenizer/tokenizer.h"

namespace llm {

std::atomic<int64_t> Sequence::next_id_{1};

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
  std::string Sequence::decode_delta_text(const Tokenizer& tokenizer) {
    const auto prefix_text =
        tokenizer.decode(sub_token_ids(prefix_offset_, read_offset_));
    const auto new_text = tokenizer.decode(sub_token_ids(prefix_offset_));
    if (new_text.size() > prefix_text.size()) {
      prefix_offset_ = read_offset_;
      read_offset_ = token_ids_.size();
      // only print the delta text
      return new_text.substr(prefix_text.size());
    }
    return "";
  }

}  // namespace llm
