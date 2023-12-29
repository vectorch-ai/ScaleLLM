#include "sequence.h"

#include <absl/strings/match.h>

#include <cstdint>
#include <string>
#include <vector>

#include "request.h"
#include "tokenizer/tokenizer.h"

namespace llm {
namespace {
// Returns whether a given `sequence` ends with `suffix`.
inline bool sequence_end_withs(const std::vector<int32_t>& sequence,
                               const std::vector<int32_t>& suffix) noexcept {
  return suffix.empty() ||
         (sequence.size() >= suffix.size() &&
          memcmp(sequence.data() + (sequence.size() - suffix.size()),
                 suffix.data(),
                 suffix.size() * sizeof(int32_t)) == 0);
}
}  // namespace

// NOLINTNEXTLINE
std::atomic<int64_t> Sequence::next_id_{1};

Sequence::Sequence(const Request& request, OnStream on_stream)
    : id_(next_id_.fetch_add(1)), request_(request), on_stream_(on_stream) {
  const auto& prompt_tokens = request_.prompt_tokens;
  // reserve enough space for the token ids to avoid reallocation
  // so that the token ids are not invalidated
  const size_t max_tokens = request_.stopping_criteria.max_tokens;
  token_ids_.reserve(max_tokens + token_ids_.size());
  token_ids_ = prompt_tokens;
  num_prompt_tokens_ = prompt_tokens.size();

  // if echo is true, set prefix_offset_ and output_offset_ to 0 to print the
  // whole sequence, otherwise set them to the length of the prompt to skip the
  // prompt.
  prefix_offset_ = request_.echo ? 0 : token_ids_.size();
  output_offset_ = request_.echo ? 0 : token_ids_.size();

  // calculate the token counts
  for (const int32_t token_id : token_ids_) {
    token_to_count_map_[token_id]++;
  }
}

bool Sequence::append_new_token_id(int32_t next_token_id) {
  if (is_finished_) {
    return false;
  }

  const auto& stopping_criteria = request_.stopping_criteria;

  // check eos and stop tokens ids first
  if (!stopping_criteria.ignore_eos_token &&
      next_token_id == stopping_criteria.eos_token_id) {
    finish_reason_ = FinishReason::STOP;
    is_finished_ = true;
    return false;
  }
  // check against stop tokens ids
  if (stopping_criteria.stop_token_ids.count(next_token_id) > 0) {
    finish_reason_ = FinishReason::STOP;
    is_finished_ = true;
    return false;
  }

  // all tokens before pos should be processed and cached.
  cache_pos_ = token_ids_.size();
  token_ids_.push_back(next_token_id);
  token_to_count_map_[next_token_id]++;

  // check against stop sequences after adding the token
  for (const auto& stop_sequence : stopping_criteria.stop_sequences) {
    if (stop_sequence.back() == next_token_id &&
        sequence_end_withs(token_ids_, stop_sequence)) {
      finish_reason_ = FinishReason::STOP;
      is_finished_ = true;
      return false;
    }
  }

  // check against max tokens
  const size_t max_new_tokens = stopping_criteria.max_tokens;
  if (max_new_tokens > 0 && num_generated_tokens() >= max_new_tokens) {
    finish_reason_ = FinishReason::LENGTH;
    is_finished_ = true;
    return false;
  }

  // return true if the sequence is not finished
  return true;
}

// TODO
bool Sequence::append_spec_token_id(int32_t spec_token_id) {
  spec_token_ids_.push_back(spec_token_id);
  ++spec_token_count_;
  return true;
}

void Sequence::clear_spec_token_ids() {
  spec_token_ids_.clear();
  spec_token_count_ = 0;
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

const SamplingParameter& Sequence::sampling_param() const {
  return request_.sampling_param;
}

}  // namespace llm
