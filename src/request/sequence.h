#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "sampling_parameter.h"
#include "stopping_criteria.h"
#include "tokenizer/tokenizer.h"

namespace llm {

// "stop" - the model hit a natural stop point or a provided stop sequence.
// "length" - the maximum number of tokens specified in the request was reached.
// "function_call" - the model called a function.
enum class FinishReason {
  VOID = 0,
  STOP = 1,
  LENGTH,
  FUNCTION_CALL,
};

// The sequence encapsulates all the necessary
// information for a sequence, including the prompt, the token ids, and the
// current position in generating tokens, etc.
class Sequence final {
 public:
  Sequence(std::string prompt,
           std::vector<int32_t> token_ids,
           const SamplingParameter* sampling_param,
           const StoppingCriteria* stopping_criteria)
      : id_(next_id_.fetch_add(1)),
        prompt_(std::move(prompt)),
        token_ids_(std::move(token_ids)),
        num_prompt_tokens_(token_ids_.size()),
        sampling_param_(sampling_param),
        stopping_criteria_(stopping_criteria) {
  }

  // get the id of the sequence
  int64_t id() const { return id_; }

  // // get token ids
  const std::vector<int32_t>& token_ids() const { return token_ids_; }

  size_t num_tokens() const { return token_ids_.size(); }

  size_t num_prompt_tokens() const { return num_prompt_tokens_; }

  const std::string& prompt() const { return prompt_; }

  const SamplingParameter& sampling_param() const { return *sampling_param_; }

  // whether the sequence is in prefill stage
  bool is_prefill() const { return cache_pos_ == 0; }

  // add a new token id to the sequence
  void append_new_token_id(int token_id) {
    // all tokens before pos should be processed and cached.
    cache_pos_ = token_ids_.size();
    token_ids_.push_back(token_id);
  }

  // add new cache blocks
  void add_blocks(const std::vector<int32_t>& new_blocks) {
    blocks_.insert(blocks_.end(), new_blocks.begin(), new_blocks.end());
  }

  // release all cache blocks
  std::vector<int32_t> release_blocks() {
    // reset the current pos to 0 so that the cache can be recomputed next time
    cache_pos_ = 0;
    return std::move(blocks_);
  }

  const std::vector<int32_t>& blocks() const { return blocks_; }

  // get the number of cache blocks to allocate for the sequence
  size_t num_blocks_to_allocate() const {
    if (is_finished_) {
      // no need to allocate more blocks for a finished sequence
      return 0;
    }
    const size_t num_slots = token_ids_.size();
    // round up to the nearest block number
    const size_t num_blocks =
        (num_slots + slots_per_block_ - 1) / slots_per_block_;
    if (num_blocks <= blocks_.size()) {
      // enough slots, don't need to allocate more
      return 0;
    }
    return num_blocks - blocks_.size();
  }

  std::vector<int32_t> slots() const {
    const size_t num_slots = token_ids_.size();
    std::vector<int32_t> slots;
    slots.reserve(num_slots);
    for (const auto& block_id : blocks_) {
      for (int32_t i = 0; i < slots_per_block_; ++i) {
        slots.push_back(block_id * slots_per_block_ + i);
        if (slots.size() == num_slots) {
          break;
        }
      }
    }
    return slots;
  }

  int32_t last_slot_id() const {
    const size_t num_slots = token_ids_.size();
    const int32_t block_offset =
        static_cast<int32_t>(num_slots - 1) % slots_per_block_;
    const int32_t last_block_id = blocks_.back();
    return last_block_id * slots_per_block_ + block_offset;
  }

  bool is_finished() const { return is_finished_; }
  FinishReason finish_reason() const { return finish_reason_; }

  // check if the sequence is finished, update the finish reason and is_finished
  // flag if finished
  bool check_stopping_creteria() {
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
  std::string decode_delta_text(const Tokenizer& tokenizer) {
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

  // the number of slots per block
  int32_t slots_per_block_ = 0;

 private:
  std::vector<int32_t> sub_token_ids(size_t start, size_t end) {
    return {token_ids_.begin() + static_cast<long>(start),
            token_ids_.begin() + static_cast<long>(end)};
  }
  std::vector<int32_t> sub_token_ids(size_t start) {
    return {token_ids_.begin() + static_cast<long>(start), token_ids_.end()};
  }

  // global unique id for the sequence
  int64_t id_ = 0;

  // prompt to generate completions for
  std::string prompt_;

  // private:
  // token ids generated from p
  std::vector<int32_t> token_ids_;

  // the length of the prompt tokens
  size_t num_prompt_tokens_ = 0;

  // sampling parameters
  const SamplingParameter* sampling_param_ = nullptr;

  const StoppingCriteria* stopping_criteria_ = nullptr;

  // the cache position.
  // all tokens before pos should be processed and cached.
  size_t cache_pos_ = 0;

  // physical block ids that hold the keys and values cache.
  std::vector<int32_t> blocks_;

  // has the sequence been finished
  bool is_finished_ = false;

  // the reason why the sequence is finished
  FinishReason finish_reason_ = FinishReason::VOID;

  // variables to keep track of output text
  size_t prefix_offset_ = 0;
  size_t read_offset_ = 0;

  // TODO: Add logits results.

  // function to call when new tokens are generated. (only for streaming)
  // std::function<void(const std::string& delta, const
  // std::string&finish_reason)> OnNewToken = nullptr;

  // id allocator for sequences
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static std::atomic<int64_t> next_id_;
};

}  // namespace llm
