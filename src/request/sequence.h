#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

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
  Sequence(std::string prompt, std::vector<int32_t> token_ids)
      : prompt_(std::move(prompt)), token_ids_(std::move(token_ids)), num_prompt_tokens_(token_ids_.size()) {
    
  }

  // whether the sequence is in prefill stage
  bool is_prefill() const { return cache_pos_ == 0; }

  // // get token ids
  const std::vector<int32_t>& token_ids() const { return token_ids_; }

  size_t num_tokens() const { return token_ids_.size(); }

  size_t num_prompt_tokens() const { return num_prompt_tokens_; }

  const std::string& prompt() const { return prompt_; }

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
    const int32_t num_slots = token_ids_.size();
    // round up to the nearest block number
    const int32_t num_blocks =
        (num_slots + slots_per_block_ - 1) / slots_per_block_;
    if (num_blocks <= blocks_.size()) {
      // enough slots, don't need to allocate more
      return 0;
    }
    return num_blocks - blocks_.size();
  }

  // const std::vector<uint32_t>& blocks() const {
  //   return blocks;
  // }

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

  bool is_finished() {
    if (is_finished_) {
      return true;
    }
    // check against stopping criterias
    const size_t generated_tokens = token_ids_.size() - num_prompt_tokens_;
    if (max_tokens_ > 0 && generated_tokens >= max_tokens_) {
      finish_reason_ = FinishReason::LENGTH;
      return is_finished_ = true;
    }
    // TODO: Add other stopping criterias

    return is_finished_;
  }

  // decode the sequence to get delta text using the tokenizer
  std::string decode_delta_text(const Tokenizer& tokenizer) {
    const auto prefix_text = tokenizer.decode(sub_token_ids(prefix_offset_, read_offset_));
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
    return {token_ids_.begin() + start, token_ids_.begin() + end};
  }
  std::vector<int32_t> sub_token_ids(size_t start) {
    return {token_ids_.begin() + start, token_ids_.end()};
  }

  // prompt to generate completions for
  std::string prompt_;

  // private:
  // token ids generated from p
  std::vector<int32_t> token_ids_;

  // the length of the prompt tokens
  size_t num_prompt_tokens_ = 0;

  // the cache position.
  // all tokens before pos should be processed and cached.
  size_t cache_pos_ = 0;

  // physical block ids that hold the keys and values cache.
  std::vector<int32_t> blocks_;

  // has the sequence been finished
  bool is_finished_ = false;

  // the reason why the sequence is finished
  FinishReason finish_reason_ = FinishReason::VOID;

  // accumulated output text
  // std::string output_text_;

  // stopping criterias
  int32_t max_tokens_ = 0;

  // end of sentence token id from tokenizer
  int32_t eos_token_id_ = 0;

  // variables to keep track of output text
  // offset of token_ids for last output text
  size_t prefix_offset_ = 0;
  // offset of token_ids for read text
  size_t read_offset_ = 0;

  // TODO: Add logits results.

  // function to call when new tokens are generated. (only for streaming)
  // std::function<void(const std::string& delta, const
  // std::string&finish_reason)> OnNewToken = nullptr;
};

}  // namespace llm
