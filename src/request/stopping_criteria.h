#pragma once

#include <cstdint>
#include <unordered_set>
#include <vector>

#include "common/slice.h"

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

// StoppingCriteria is used to specify stopping criterias for a
// request/sequence.
struct StoppingCriteria {
 public:
  FinishReason check_finished(const Slice<int32_t>& token_ids,
                              size_t num_prompt_tokens) const;

  // private:

  // maximum number of generated tokens
  size_t max_tokens = 0;

  // end of sentence token id from tokenizer
  int32_t eos_token_id = 0;

  // whether to ignore eos token when checking stopping criterias
  bool ignore_eos_token = false;

  // stop token ids
  std::unordered_set<int32_t> stop_token_ids;

  // stop sequences
  std::vector<std::vector<int32_t>> stop_sequences;

  // max context length
  size_t max_context_length = 0;
};

}  // namespace llm
