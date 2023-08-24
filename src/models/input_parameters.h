#pragma once

#include <torch/torch.h>

namespace llm {
// input parameters for the model that encapsulates all the necessary
// information required to process a batch efficiently, mainly for
// cache management.
struct InputParameters {
  // number of prompt tokens in the batch.
  int64_t num_prompt_tokens;

  // cumulative sequence length of each sequence.
  // used in prefill stage to determine the token range for each sequence
  // [num_prompt_seq + 1]
  // for example: 3 sequences with length 2, 3, 4, 
  // the cu_seq_lens is [0, 2, 5, 9]
  torch::Tensor cu_seq_lens;

  // maximum sequence length in the prompt.
  int32_t max_seq_len; 

  // logical cache slot for each token.
  // used to store kv-cache to right slot/block
  // [num_prompt_tokens] IntTensor
  torch::Tensor slot_ids;

  // block ids for each sequence.
  // used in generate stage to fetch cached key-value.
  // [num_generate_seq, max_num_blocks] IntTensor
  torch::Tensor block_tables;

  // number of tokens for each sequence.
  // used in generate stage to determine the range of cache to fetch
  // [num_generate_seq] IntTensor
  torch::Tensor context_lens;

  // the index of the last token of each sequence in the batch.
  torch::Tensor sample_idx;
};

// output parameters for the model that encapsulates all the necessary
// output information. The output parameters should be as small as possible
// to avoid transferring large tensors between host and device.
struct OutputParameters {

  torch::Tensor logits;

  // the index of the last token of each sequence in the batch.
  torch::Tensor output_tokens;
};

}  // namespace llm
