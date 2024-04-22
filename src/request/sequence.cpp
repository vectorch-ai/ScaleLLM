#include "sequence.h"

#include <absl/strings/match.h>

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#include "common/slice.h"
#include "tokenizer/tokenizer.h"

DEFINE_int32(num_speculative_tokens, 0, "number of speculative tokens");

namespace llm {

// NOLINTNEXTLINE
std::atomic<int64_t> Sequence::next_id_{1};

Sequence::Sequence(const std::string_view& prompt,
                   const std::vector<int32_t>& prompt_token_ids,
                   const SamplingParameter& sampling_param,
                   const StoppingCriteria& stopping_criteria,
                   bool echo,
                   OnDelta on_delta)
    : id_(next_id_.fetch_add(1)),
      sampling_param_(sampling_param),
      stopping_criteria_(stopping_criteria),
      decoder_(prompt,
               prompt_token_ids.size(),
               echo,
               sampling_param.skip_special_tokens),
      num_kv_cache_tokens_(static_cast<size_t>(EngineType::COUNT), 0),
      on_delta_(on_delta) {
  CHECK(!prompt_token_ids.empty()) << "empty prompt token ids";

  // allocate space for the token ids and add the prompt tokens
  const size_t max_tokens = stopping_criteria.max_tokens +
                            prompt_token_ids.size() +
                            FLAGS_num_speculative_tokens + /*bouns_token*/ 1;
  token_ids_.resize(max_tokens);
  for (const auto token_id : prompt_token_ids) {
    token_ids_[num_tokens_++] = token_id;
    token_to_count_map_[token_id]++;
  }
  num_prompt_tokens_ = num_tokens_;
}

void Sequence::append_new_token_id(int32_t next_token_id) {
  CHECK(num_tokens_ < token_ids_.size())
      << "exceed the token capacity of the sequence";
  CHECK(!is_finished_) << "cannot append token to a finished sequence";
  CHECK(!is_prefill_stage()) << "cannot append token to a prefill sequence";

  // append the token id and update the token count
  token_ids_[num_tokens_++] = next_token_id;
  ++token_to_count_map_[next_token_id];

  // invalidate the finish status once a new token is appended
  finish_status_invalidated_ = true;
}

size_t Sequence::validate_token_ids(const Slice<int64_t>& accpeted_token_ids) {
  const size_t len = accpeted_token_ids.size();
  CHECK_GT(num_tokens_, len) << "accepted tokens exceed the sequence length";

  // validate the accepted tokens with draft tokens, stop at the first mismatch
  const size_t start_idx = num_tokens_ - len;
  size_t accpeted_len = 0;
  for (size_t i = 0; i < len; ++i) {
    const size_t cur_idx = start_idx + i;
    const int32_t draft_token_id = token_ids_[cur_idx];
    const int32_t target_token_id = static_cast<int32_t>(accpeted_token_ids[i]);

    // stop at first rejected token id
    if (target_token_id == -1) {
      num_tokens_ = cur_idx;
      break;
    }

    ++accpeted_len;
    if (target_token_id != draft_token_id) {
      // overwrite the token id with the accepted token id
      token_ids_[cur_idx] = target_token_id;
      // update the token count
      --token_to_count_map_[draft_token_id];
      ++token_to_count_map_[target_token_id];
    }

    // check if sequence is finished
    const Slice<int32_t> token_ids(token_ids_, cur_idx + 1);
    auto finish_reason =
        stopping_criteria_.check_finished(token_ids, num_prompt_tokens_);
    if (finish_reason != FinishReason::NONE) {
      finish_reason_ = finish_reason;
      is_finished_ = true;
      // update num tokens, including current token
      num_tokens_ = cur_idx + 1;
      break;
    }
  }

  // adjust the token count for remaining discarded tokens
  for (size_t i = accpeted_len; i < len; ++i) {
    const auto token_id = token_ids_[start_idx + i];
    --token_to_count_map_[token_id];
  }

  // adjust kv cache position
  // num_tokens must be at least one more than num_kv_cache_tokens
  for (auto& num_kv_cache_tokens : num_kv_cache_tokens_) {
    num_kv_cache_tokens = std::min(num_kv_cache_tokens, num_tokens_ - 1);
  }

  // the finish status is valid after the validation
  finish_status_invalidated_ = false;
  return accpeted_len;
}

// decode the sequence to get delta text using the tokenizer
std::string Sequence::decode_delta_text(const Slice<int32_t>& token_ids,
                                        const Tokenizer& tokenizer) {
  return decoder_.decode(token_ids, tokenizer);
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
  size_t num_shared_tokens = shared_blocks.size() * block_size;
  blocks_.insert(blocks_.end(), shared_blocks.begin(), shared_blocks.end());

  // It is possible that num_shared_tokens == num_prompt_tokens_, indicating
  // that the exact same prompt has been received again. In this case, it
  // becomes necessary to adjust the kv cache position to the previous token,
  // allowing the model proceed. While the shared blocks should be immutable
  // ideally, but it remains safe to regenerate the kv cache in this context,
  // given the utiliztion of the exact same token.
  if (num_shared_tokens == num_prompt_tokens_) {
    num_shared_tokens -= 1;
  }
  CHECK(num_shared_tokens < num_prompt_tokens_);
  // update the kv cache position
  std::fill(num_kv_cache_tokens_.begin(),
            num_kv_cache_tokens_.end(),
            num_shared_tokens);
}

// release all cache blocks
void Sequence::release_blocks() {
  // reset the kv cache position to 0
  std::fill(num_kv_cache_tokens_.begin(), num_kv_cache_tokens_.end(), 0);
  blocks_.clear();
}

size_t Sequence::kv_cache_capacity() const {
  if (blocks_.empty()) {
    return 0;
  }
  // all blocks have the same size
  const size_t block_size = blocks_[0].size();
  return blocks_.size() * block_size;
}

std::vector<int32_t> Sequence::kv_cache_slots(int32_t pos_start,
                                              int32_t pos_end) const {
  CHECK(!blocks_.empty()) << "no cache blocks available";

  std::vector<int32_t> slots;
  slots.reserve(pos_end - pos_start);

  const size_t block_size = blocks_[0].size();
  for (int32_t i = pos_start; i < pos_end; ++i) {
    const int32_t block_id = blocks_[i / block_size].id();
    const int32_t block_offset = i % block_size;
    slots.push_back(block_id * block_size + block_offset);
  }
  return slots;
}

void Sequence::stream_delta(const SequenceDeltaOutput& output) {
  if (on_delta_) {
    if (!on_delta_(output)) {
      // cancel the sequence if the callback returns false
      is_cancelled_.store(true, std::memory_order_relaxed);
    }
  }
}

bool Sequence::is_cancelled() const {
  return is_cancelled_.load(std::memory_order_relaxed);
}

bool Sequence::is_finished() const {
  // return the cached finish status
  if (!finish_status_invalidated_) {
    return is_finished_;
  }

  // reset the finish status invalidation flag
  finish_status_invalidated_ = false;

  auto finish_reason =
      stopping_criteria_.check_finished(token_ids(), num_prompt_tokens_);
  if (finish_reason != FinishReason::NONE) {
    finish_reason_ = finish_reason;
    is_finished_ = true;
    return true;
  }
  return false;
}

}  // namespace llm
