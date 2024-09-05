#pragma once

#include <torch/torch.h>

#include "common/tensor_helper.h"

namespace llm {
// input parameters for the model that encapsulates all the necessary
// information required to process a batch efficiently, mainly for
// self-attention and kv-cache.
struct InputParameters {
  InputParameters to(const torch::Device& device) const {
    InputParameters params;
    // copy scalar values
    params.empty_kv_cache = empty_kv_cache;
    params.num_sequences = num_sequences;
    params.kv_max_seq_len = kv_max_seq_len;
    params.q_max_seq_len = q_max_seq_len;

    // all tensors should be on the same device
    params.kv_cu_seq_lens = safe_to(kv_cu_seq_lens, device);
    params.q_cu_seq_lens = safe_to(q_cu_seq_lens, device);

    params.new_cache_slots = safe_to(new_cache_slots, device);
    params.block_tables = safe_to(block_tables, device);
    params.cu_block_lens = safe_to(cu_block_lens, device);
    return params;
  }

  // whether the kv-cache is empty for all sequences.
  bool empty_kv_cache = true;

  // total number of sequences in the batch
  int32_t num_sequences = 0;

  // cumulative sequence length of each sequence
  // used to determine the token range for each sequence
  // IntTensor: [n_seq + 1]
  // for example: 3 sequences with length 2, 3, 4,
  // the cu_seq_lens is [0, 2, 5, 9]
  torch::Tensor q_cu_seq_lens;   // query len
  torch::Tensor kv_cu_seq_lens;  //  kv len: (tokens in cache + new tokens)

  // maximum sequence length for query and kv.
  // used to help dispatch, choosing the right kernel based on the lenght
  int32_t kv_max_seq_len = 0;  // kv seq len
  int32_t q_max_seq_len = 0;   // query seq len

  // logical cache slot for each *new* token.
  // used to store kv-cache to right slot/block
  // IntTensor: [n_tokens]
  torch::Tensor new_cache_slots;

  // block ids for each sequence, flattend into 1D tensor.
  // IntTensor: [n_blocks]
  torch::Tensor block_tables;
  // cumulative block length for each sequence.
  // IntTensor: [n_seq + 1]
  torch::Tensor cu_block_lens;
};

}  // namespace llm
