#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "common/slice.h"
#include "incremental_decoder.h"
#include "memory/block.h"
#include "output.h"
#include "sampling/parameters.h"
#include "stopping_criteria.h"
#include "tokenizer/tokenizer.h"

namespace llm {

using OnDelta = std::function<bool(const SequenceOutput& output)>;

// The sequence is shared between LLM and SSM for speculative decoding, and
// it's possible that the numbers of tokens in kv cache are out of sync.
// Specifying the engine type to ensure accurate updating of the the number
// tokens in kv cache separately for LLM and SSM.
enum class EngineType : int8_t {
  // LLM engine
  LLM = 0,
  // SSM engine
  SSM = 1,
  // total number of engines
  COUNT = 2,
};

// The sequence encapsulates all the necessary
// information for a sequence, including the prompt, the token ids, and the
// current position in generating tokens, etc.
class Sequence final {
 public:
  struct Options {
    // the sampling parameters for the sequence
    SamplingParameter sampling_param;

    // the stopping criteria for the sequence
    StoppingCriteria stopping_criteria;

    // whether to skip special tokens when decoding the output
    bool skip_special_tokens = true;

    // whether to echo the prompt tokens back
    bool echo = false;

    // the callback function to call when new tokens are generated
    OnDelta on_delta = nullptr;
  };

  Sequence(const std::string_view& prompt,
           const std::vector<int32_t>& prompt_token_ids,
           size_t capacity,
           const Options& option);

  // get the id of the sequence
  int64_t id() const { return id_; }

  // get the index of the sequence in the request
  size_t index() const { return index_; }

  // get token ids
  Slice<int32_t> token_ids() const { return {token_ids_, num_tokens_}; }

  // get token ids to count map
  const std::unordered_map<int32_t, int32_t>& token_to_count_map() const {
    return token_to_count_map_;
  }

  // get the total number of tokens
  size_t num_tokens() const { return num_tokens_; }

  // get the number of prompt tokens
  size_t num_prompt_tokens() const { return num_prompt_tokens_; }

  // get the number of generated tokens
  // returns 0 if still in prefill stage
  size_t num_generated_tokens() const {
    return num_tokens_ - num_prompt_tokens_;
  }

  // get token ids in kv cache
  Slice<int32_t> tokens_in_kv_cache() const {
    // it is a little bit tricky to get the tokens in kv cache for speculative
    // decoding where the number of tokens in kv cache may be out of sync by at
    // most 1 between LLM and SSM.
    const size_t ssm_kv_cache_size = num_kv_cache_tokens(EngineType::SSM);
    const size_t llm_kv_cache_size = num_kv_cache_tokens(EngineType::LLM);
    CHECK_GE(llm_kv_cache_size, ssm_kv_cache_size);
    const size_t diff = llm_kv_cache_size - ssm_kv_cache_size;
    // at most one token difference between LLM and SSM for speculative decoding
    const size_t kv_cache_size =
        diff <= 1 ? ssm_kv_cache_size : llm_kv_cache_size;
    return {token_ids_, kv_cache_size};
  }

  // get the number of tokens in the kvcache
  size_t num_kv_cache_tokens() const {
    return num_kv_cache_tokens_[engine_type_];
  }

  size_t num_kv_cache_tokens(EngineType engine_type) const {
    CHECK(engine_type < EngineType::COUNT) << "Invalid engine type";
    return num_kv_cache_tokens_[static_cast<size_t>(engine_type)];
  }

  // get the capacity of the kv cache allocated
  size_t kv_cache_capacity() const;

  // generate the kv cache slots for the position range [pos_start, pos_end)
  std::vector<int32_t> kv_cache_slots(int32_t pos_start, int32_t pos_end) const;

  // get the number of tokens to process
  size_t num_tokens_to_process() const {
    return num_tokens() - num_kv_cache_tokens();
  }

  // check if the sequence is in prefill stage
  bool is_prefill_stage() const {
    return num_kv_cache_tokens() < num_prompt_tokens();
  }

  // add a new token id to the sequence and update the count
  // the token would be discarded if the sequence is still in prefill stage
  void append_token(int32_t token_id);

  // validate draft tokens with accepted tokens for speculative decoding
  // N.B. take int64_t as input to be compatible with torch::Tensor
  // returns the number of accepted tokens, including the resampled token
  size_t validate_tokens(const Slice<int64_t>& accpeted_token_ids);

  // add new cache blocks
  void append_block(const Block& new_block) {
    return append_blocks({new_block});
  }
  void append_blocks(const std::vector<Block>& new_blocks);

  // append shared cache blocks from prefix cache
  void append_shared_blocks(const std::vector<Block>& shared_blocks);

  // release all cache blocks
  void release_blocks();

  // returns allocated cache blocks
  Slice<Block> blocks() const { return blocks_; }

  // get the number of blocks
  size_t num_blocks() const { return blocks_.size(); }

  // get the reason why the sequence is finished
  FinishReason finish_reason() const { return finish_reason_; }

  // decode the tokens till end to get delta text using the tokenizer
  // not thread safe
  std::string decode_delta_text(const Slice<int32_t>& token_ids,
                                const Tokenizer& tokenizer);

  // get the offset of output tokens
  size_t output_offset() const { return decoder_.output_offset(); }

  // check if streaming is enabled
  bool is_streaming() const { return options_.on_delta != nullptr; }

  // stream the delta output to the client
  // cancel the sequence if the callback returns false
  void stream_delta(const SequenceOutput& output);

  // check if the sequence is cancelled
  bool is_cancelled() const;

  // check finish status, use cached value if not invalidated
  bool is_finished() const;

  // set engine type this sequence is used for
  void set_engine_type(EngineType engine_type) {
    CHECK(engine_type < EngineType::COUNT) << "Invalid engine type.";
    engine_type_ = static_cast<size_t>(engine_type);
  }

  // commit the kv cache by n tokens
  void commit_kv_cache(size_t size) {
    size_t& num_kv_cache_tokens = num_kv_cache_tokens_[engine_type_];
    CHECK(num_kv_cache_tokens + size <= kv_cache_capacity());
    num_kv_cache_tokens += size;
  }

  // get the sampling parameters
  const SamplingParameter* sampling_param() const {
    return &options_.sampling_param;
  }

  // get the stopping criteria
  const StoppingCriteria* stopping_criteria() const {
    return &options_.stopping_criteria;
  }

 private:
  // global unique id for the sequence
  const int64_t id_;

  // the index of the sequence in the request
  size_t index_ = 0;

  // options for the sequence
  Options options_;

  // incremental decoder to decode the tokens
  IncrementalDecoder decoder_;

  // token ids generated for the sequence
  std::vector<int32_t> token_ids_;

  // number of tokens in the sequence
  size_t num_tokens_ = 0;

  // the count of each token id
  std::unordered_map<int32_t, int32_t> token_to_count_map_;

  // the length of the prompt tokens
  size_t num_prompt_tokens_ = 0;

  // number of tokens in kv cache
  std::vector<size_t> num_kv_cache_tokens_;
  // current using engine type
  size_t engine_type_ = 0;

  // physical blocks that hold the kv cache.
  std::vector<Block> blocks_;

  // is the sequence cancelled
  std::atomic_bool is_cancelled_{false};

  // is the sequence finished
  mutable bool is_finished_ = false;

  // is the finish status invalidated
  mutable bool finish_status_invalidated_ = true;

  // the reason why the sequence is finished
  mutable FinishReason finish_reason_ = FinishReason::NONE;

  // id allocator for sequences
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static std::atomic<int64_t> next_id_;
};

}  // namespace llm
