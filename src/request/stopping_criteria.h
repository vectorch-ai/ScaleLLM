#pragma once

#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

namespace llm {

// StoppingCriteria is used to specify stopping criterias for a
// request/sequence.
struct StoppingCriteria {
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
};

}  // namespace llm
