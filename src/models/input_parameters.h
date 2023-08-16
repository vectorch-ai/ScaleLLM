#pragma once

#include <torch/torch.h>

namespace llm {
// input parameters for the model that encapsulates all the necessary
// information required to process a batch efficiently, mainly for
// cache management.
struct InputParameters {
  // cumulative sequence length of each sequence.
  // used in prefill stage to determine the token range for each sequence
  // [num_prompt_seq + 1]
  // for example: 3 sequences with length 2, 3, 4, the cu_seq_lens is [0, 2, 5,
  // 9]
  std::vector<int64_t> cu_seq_lens;

  // logical cache slot for each token.
  // used to store kv-cache to right slot/block
  // [num_prompt_tokens]
  torch::Tensor slots;

  // block ids for each sequence.
  // used in generate stage to fetch cached key-value.
  // [num_generate_seq, max_num_blocks]
  torch::Tensor block_tables;

  // number of tokens for each sequence.
  // used in generate stage to determine the range of cache to fetch
  // [num_generate_seq]
  torch::Tensor context_lens;
};

}  // namespace llm
