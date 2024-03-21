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

Sequence::Sequence(const SamplingParameter& sampling_param,
                   const StoppingCriteria& stopping_criteria,
                   const std::vector<int32_t>& token_ids,
                   bool echo,
                   OnStream on_stream)
    : id_(next_id_.fetch_add(1)),
      sampling_param_(sampling_param),
      stopping_criteria_(stopping_criteria),
      token_ids_(token_ids),
      on_stream_(on_stream) {
  num_prompt_tokens_ = token_ids_.size();
  // reserve enough space for the token ids to avoid reallocation
  // so that the token ids are not invalidated
  const size_t max_tokens = stopping_criteria.max_tokens;
  token_ids_.reserve(max_tokens + num_prompt_tokens_);

  // if echo is true, set prefix_offset_ and output_offset_ to 0 to print the
  // whole sequence, otherwise set them to the length of the prompt to skip the
  // prompt.
  prefix_offset_ = echo ? 0 : token_ids_.size();
  output_offset_ = echo ? 0 : token_ids_.size();

  // calculate the token counts
  for (const int32_t token_id : token_ids_) {
    token_to_count_map_[token_id]++;
  }
}

bool Sequence::append_new_token_id(int32_t next_token_id) {
  if (is_finished_) {
    return false;
  }

  // check eos and stop tokens ids first
  if (!stopping_criteria_.ignore_eos_token &&
      next_token_id == stopping_criteria_.eos_token_id) {
    finish_reason_ = FinishReason::STOP;
    is_finished_ = true;
    return false;
  }
  // check against stop tokens ids
  if (stopping_criteria_.stop_token_ids.count(next_token_id) > 0) {
    finish_reason_ = FinishReason::STOP;
    is_finished_ = true;
    return false;
  }

  // all tokens before pos should be processed and cached.
  kv_cache_pos_ = token_ids_.size();
  token_ids_.push_back(next_token_id);
  token_to_count_map_[next_token_id]++;

  // check against stop sequences after adding the token
  for (const auto& stop_sequence : stopping_criteria_.stop_sequences) {
    if (stop_sequence.back() == next_token_id &&
        sequence_end_withs(token_ids_, stop_sequence)) {
      finish_reason_ = FinishReason::STOP;
      is_finished_ = true;
      return false;
    }
  }

  // check against max tokens
  const size_t max_new_tokens = stopping_criteria_.max_tokens;
  if (max_new_tokens > 0 && num_generated_tokens() >= max_new_tokens) {
    finish_reason_ = FinishReason::LENGTH;
    is_finished_ = true;
    return false;
  }

  // return true if the sequence is not finished
  return true;
}

void Sequence::append_spec_token_id(int32_t spec_token_id) {
  spec_token_ids_.push_back(spec_token_id);
}

void Sequence::update_valid_token_ids(const int64_t* valid_ids) {
  // reset finished flags
  finish_reason_ = FinishReason::NONE;
  is_finished_ = false;

  size_t idx = 0;
  for (; idx < spec_token_ids_.size(); ++idx) {
    if (valid_ids[idx] != spec_token_ids_[idx]) {
      // find first invalid token id (idx)
      // 1. clear invalid token counts
      for (int64_t i = idx; i < spec_token_ids_.size(); ++i) {
        token_to_count_map_[spec_token_ids_[i]]--;
      }
      // 2. erase invalid tokens
      token_ids_.erase(token_ids_.end() - spec_token_ids_.size() + idx,
                       token_ids_.end());
      break;
    }
  }
  // clear spec token ids
  spec_token_ids_.clear();

  // append new valid id
  append_new_token_id(valid_ids[idx]);
}

// decode the sequence to get delta text using the tokenizer
std::string Sequence::decode_delta_text(size_t end,
                                        const Tokenizer& tokenizer) {
  const auto tokens = token_ids();
  const auto prefix_text =
      tokenizer.decode(tokens.slice(prefix_offset_, output_offset_));
  const auto new_text = tokenizer.decode(tokens.slice(prefix_offset_, end));
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

size_t Sequence::num_generated_tokens() const {
  const size_t n_tokens = num_tokens();
  const size_t n_prompt_tokens = num_prompt_tokens();
  return (n_tokens <= n_prompt_tokens) ? 0 : n_tokens - n_prompt_tokens;
}

void Sequence::append_blocks(const std::vector<Block>& new_blocks) {
  blocks_.insert(blocks_.end(), new_blocks.begin(), new_blocks.end());
}

// append shared cache blocks from prefix cache
void Sequence::append_shared_blocks(const std::vector<Block>& shared_blocks) {
  CHECK(blocks_.empty()) << "shared blocks should be appended before any "
                            "other blocks";
  if (shared_blocks.empty()) {
    return;
  }
  // update the kv cache position
  const size_t block_size = shared_blocks[0].size();
  kv_cache_pos_ = shared_blocks.size() * block_size;
  blocks_.insert(blocks_.end(), shared_blocks.begin(), shared_blocks.end());
}

// release all cache blocks
void Sequence::release_blocks() {
  // reset the current pos to 0
  kv_cache_pos_ = 0;
  blocks_.clear();
}

void Sequence::stream_delta(const std::string& delta, FinishReason reason) {
  if (on_stream_) {
    if (!on_stream_(delta, reason)) {
      // failed to stream the delta, cancel the sequence
      set_cancelled();
    }
  }
}

}  // namespace llm
