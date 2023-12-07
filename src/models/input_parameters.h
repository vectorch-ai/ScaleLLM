#pragma once

#include <request/request.h>
#include <torch/torch.h>

namespace llm {
// input parameters for the model that encapsulates all the necessary
// information required to process a batch efficiently, mainly for
// self-attention and kv-cache.
struct InputParameters {
  InputParameters to(const torch::Device& device) const {
    InputParameters params;
    params.num_prompt_tokens = num_prompt_tokens;
    params.max_seq_len = max_seq_len;
    params.max_context_len = max_context_len;

    // all tensors should be on the same device as model
    params.cu_seq_lens = cu_seq_lens.to(device);
    params.slot_ids = slot_ids.to(device);
    params.block_tables = block_tables.to(device);
    params.context_lens = context_lens.to(device);
    params.last_token_idxes = last_token_idxes.to(device);
    params.token_ids = token_ids.to(device);
    params.token_counts = token_counts.to(device);
    params.token_ids_lens = token_ids_lens.to(device);
    return params;
  }

  // *******************************************************
  // ******   parameters only for prefill stage   *******
  // *******************************************************

  // total number of tokens in prompt sequences.
  int64_t num_prompt_tokens = 0;

  // cumulative sequence length of each sequence.
  // used in prefill stage to determine the token range for each sequence
  // [num_prompt_seq + 1]
  // for example: 3 sequences with length 2, 3, 4,
  // the cu_seq_lens is [0, 2, 5, 9] IntTensor
  torch::Tensor cu_seq_lens;

  // maximum sequence length for prompt sequences.
  int32_t max_seq_len = 0;

  // *******************************************************
  // ******   parameters only for decode stage   *******
  // *******************************************************

  // logical cache slot for each token.
  // used to store kv-cache to right slot/block
  // [num_prompt_tokens] IntTensor
  torch::Tensor slot_ids;

  // block ids for each sequence.
  // used in decode stage to fetch cached key-value.
  // [num_decode_seq, max_num_blocks] IntTensor
  torch::Tensor block_tables;

  // the maximum context len for decode sequence.
  int32_t max_context_len = 0;

  // number of tokens for each sequence.
  // used in decode stage to determine the range of cache to fetch
  // [num_decode_seq] IntTensor
  torch::Tensor context_lens;

  // *******************************************************
  // *****  parameters for all sequence in the batch  ******
  // *******************************************************

  // the index of the last token of each sequence in the tokens.
  // for prompt sequence, it is the index of last token in the prompt.
  // for decode sequence, it is the index of the token. (only one token)
  // IntTensor
  torch::Tensor last_token_idxes;

  // the unique token ids of each sequence in the batch.
  // [num_seq, max_unique_tokens] LongTensor
  torch::Tensor token_ids;

  // the count of each token in each sequence.
  // [num_seq, max_unique_tokens] IntTensor
  torch::Tensor token_counts;

  // the number of unique tokens in each sequence.
  // [num_seq] IntTensor
  torch::Tensor token_ids_lens;
};

}  // namespace llm
