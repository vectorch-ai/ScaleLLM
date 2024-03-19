#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "common/slice.h"
#include "memory/block.h"
#include "sampling_parameters.h"
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
  Sequence(const SamplingParameter& sampling_param_,
           const StoppingCriteria& stopping_criteria_,
           const std::vector<int32_t>& token_ids,
           bool echo,
           OnStream on_stream);

  // get the id of the sequence
  int64_t id() const { return id_; }

  // get token ids
  Slice<int32_t> token_ids() const { return {token_ids_}; }

  // get token ids to count map
  const std::unordered_map<int32_t, int32_t>& token_to_count_map() const {
    return token_to_count_map_;
  }

  // get the total number of tokens
  size_t num_tokens() const { return token_ids_.size(); }

  // get the number of prompt tokens
  size_t num_prompt_tokens() const { return num_prompt_tokens_; }

  // get the number of generated tokens
  // returns 0 if still in prefill stage
  size_t num_generated_tokens() const;

  // get token ids in kv cache
  Slice<int32_t> tokens_in_kv_cache() const {
    return {token_ids_, kv_cache_pos_};
  }

  // get the number of tokens in the kvcache
  size_t num_tokens_in_kv_cache() const { return kv_cache_pos_; }

  // whether the sequence is in prefill stage, no kv cache has been generated
  bool is_prefill() const { return kv_cache_pos_ == 0; }

  // add a new token id to the sequence and check if the sequence is finished.
  // returns false if the sequence is finished.
  bool append_new_token_id(int32_t next_token_id);

  // append speculate token id
  void append_spec_token_id(int32_t spec_token_id);

  // update valid token ids
  void update_valid_token_ids(const int64_t* ids);

  // add new cache blocks
  void append_blocks(const std::vector<Block>& new_blocks);

  // append shared cache blocks from prefix cache
  void append_shared_blocks(const std::vector<Block>& shared_blocks);

  // release all cache blocks
  void release_blocks();

  // returns allocated cache blocks
  Slice<Block> blocks() const { return {blocks_}; }

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
  // not thread safe
  std::string decode_delta_text(size_t end, const Tokenizer& tokenizer);

  // check if streaming is enabled
  bool is_streaming() const { return on_stream_ != nullptr; }

  // stream the delta text to the client
  void stream_delta(const std::string& delta, FinishReason reason);

  // get the offset of output tokens
  size_t output_offset() const { return output_offset_; }

  // get the sampling parameters
  const SamplingParameter& sampling_param() const { return sampling_param_; }

  // get the stopping criteria
  const StoppingCriteria& stopping_criteria() const {
    return stopping_criteria_;
  }

 private:
  std::vector<int32_t> sub_token_ids(size_t start, size_t end) {
    return {token_ids_.begin() + static_cast<long>(start),
            token_ids_.begin() + static_cast<long>(end)};
  }

  // global unique id for the sequence
  const int64_t id_;

  // the sampling parameters
  const SamplingParameter& sampling_param_;

  // the stopping criteria
  const StoppingCriteria& stopping_criteria_;

  // token ids generated for the sequence
  std::vector<int32_t> token_ids_;

  // the count of each token id
  std::unordered_map<int32_t, int32_t> token_to_count_map_;

  // the length of the prompt tokens
  size_t num_prompt_tokens_ = 0;

  // the cache position.
  // all tokens before pos should already be in the kv cache.
  size_t kv_cache_pos_ = 0;

  // physical blocks that hold the kv cache.
  std::vector<Block> blocks_;

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

  // speculative decoding tokens
  std::vector<int32_t> spec_token_ids_;
};

}  // namespace llm
