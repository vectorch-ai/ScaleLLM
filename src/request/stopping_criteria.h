#pragma once

#include <cstdint>
#include <vector>
#include <string>

namespace llm {

// StoppingCriteria is used to specify stopping criterias for a request/sequence.
struct StoppingCriteria {
  // maximum number of generated tokens
  size_t max_tokens = 0;

  // end of sentence token id from tokenizer
  int32_t eos_token_id = 0;

  // whether to ignore eos token when checking stopping criterias
  bool ignore_eos_token = false;

  // stop sequences
  // std::vector<std::string> stop_sequences;

};

}  // namespace llm
