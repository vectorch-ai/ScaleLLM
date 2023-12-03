#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
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
  NONE = 0,
  STOP = 1,
  LENGTH,
  FUNCTION_CALL,
};

using OnStream =
    std::function<bool(const std::string& delta, FinishReason reason)>;

// The sequence encapsulates all the necessary
// information for a sequence, including the prompt, the token ids, and the
// current position in generating tokens, etc.
class Sequence final {
 public:
  Sequence(std::string prompt,
           std::vector<int32_t> token_ids,
           const SamplingParameter* sampling_param,
           const StoppingCriteria* stopping_criteria,
           OnStream on_stream,
           bool echo);

  // get the id of the sequence
  int64_t id() const { return id_; }

  // get token ids
  const std::vector<int32_t>& token_ids() const { return token_ids_; }

  // get the total number of tokens
  size_t num_tokens() const { return token_ids_.size(); }

  // get the number of prompt tokens
  size_t num_prompt_tokens() const { return num_prompt_tokens_; }

  // get the prompt string
  const std::string& prompt() const { return prompt_; }

  // get the sampling parameters
  const SamplingParameter& sampling_param() const { return *sampling_param_; }

  // whether the sequence is in prefill stage, no kv cache has been generated
  bool is_prefill() const { return cache_pos_ == 0; }

  // add a new token id to the sequence and check if the sequence is finished.
  // returns false if the sequence is finished.
  bool append_new_token_id(int32_t next_token_id);

  // add new cache blocks
  void append_blocks(const std::vector<int32_t>& new_blocks) {
    blocks_.insert(blocks_.end(), new_blocks.begin(), new_blocks.end());
  }

  // release all cache blocks
  std::vector<int32_t> release_blocks() {
    // reset the current pos to 0 so that the cache can be recomputed next time
    cache_pos_ = 0;
    return std::move(blocks_);
  }

  // returns allocated cache blocks
  const std::vector<int32_t>& blocks() const { return blocks_; }

  // get the number of blocks
  size_t num_blocks() const { return blocks_.size(); }

  // check if the sequence is finished
  bool is_finished() const { return is_cancelled() || is_finished_; }

  // check if the sequence is cancelled
  bool is_cancelled() const {
    return is_cancelled_.load(std::memory_order_relaxed);
  }

  // cancel the sequence
  void set_cancelled() { is_cancelled_.store(true, std::memory_order_relaxed); }

  // get the reason why the sequence is finished
  FinishReason finish_reason() const { return finish_reason_; }

  // decode the tokens till end to get delta text using the tokenizer
  std::string decode_delta_text(size_t end, const Tokenizer& tokenizer);

  // check if streaming is enabled
  bool is_streaming() const { return on_stream_ != nullptr; }

  // stream the delta text to the client
  void stream_delta(const std::string& delta, FinishReason reason) {
    if (on_stream_) {
      if (!on_stream_(delta, reason)) {
        // failed to stream the delta, cancel the sequence
        set_cancelled();
      }
    }
  }

  // get the offset of output text
  size_t output_offset() const { return output_offset_; }

 private:
  std::vector<int32_t> sub_token_ids(size_t start, size_t end) {
    return {token_ids_.begin() + static_cast<long>(start),
            token_ids_.begin() + static_cast<long>(end)};
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

  // has the sequence been cancelled by client, e.g. timeout, rpc error, etc.
  // use a atomic bool since it can be accessed by multiple threads.
  std::atomic<bool> is_cancelled_{false};

  // the reason why the sequence is finished
  FinishReason finish_reason_ = FinishReason::NONE;

  // variables to keep track of output text, should be accessed by single thread
  // prefix offset is used to defeat cleanup algorithms in the decode which
  // decide to add a space or not based on surrounding tokens.
  size_t prefix_offset_ = 0;
  // all tokens before output_offset_ have been streamed to the client
  size_t output_offset_ = 0;

  // function to call when new tokens are generated. (only for streaming)
  OnStream on_stream_;

  // TODO: Add logits results.

  // id allocator for sequences
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static std::atomic<int64_t> next_id_;
};

}  // namespace llm
