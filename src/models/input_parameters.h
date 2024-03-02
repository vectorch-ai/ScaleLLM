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
    params.kv_max_seq_len = kv_max_seq_len;
    params.q_max_seq_len = q_max_seq_len;

    // all tensors should be on the same device as model
    auto safe_to = [](const torch::Tensor& t, const torch::Device& device) {
      return t.defined() ? t.to(device) : t;
    };
    params.kv_cu_seq_lens = safe_to(kv_cu_seq_lens, device);
    params.q_cu_seq_lens = safe_to(q_cu_seq_lens, device);

    params.new_cache_slots = safe_to(new_cache_slots, device);
    params.block_tables = safe_to(block_tables, device);
    params.last_token_idxes = safe_to(last_token_idxes, device);
    params.token_ids = safe_to(token_ids, device);
    params.token_counts = safe_to(token_counts, device);
    params.token_ids_lens = safe_to(token_ids_lens, device);
    return params;
  }

  // *******************************************************
  // ******       parameters for attention           *******
  // *******************************************************

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

  // block ids for each sequence.
  // used in attention kernel to fetch cached key-value.
  // IntTensor: [n_seq, max_n_blocks]
  torch::Tensor block_tables;

  // *******************************************************
  // *****  parameters for all sequence in the batch  ******
  // *******************************************************

  // the index of the last token of each sequence in the tokens.
  // IntTensor: [n_seqs]
  torch::Tensor last_token_idxes;

  // the unique token id and count of each sequence in the batch.
  // LongTensor: [n_seqs, max_unique_tokens]
  torch::Tensor token_ids;
  torch::Tensor token_counts;  // IntTensor

  // the number of unique tokens in each sequence.
  // IntTensor: [n_seqs]
  torch::Tensor token_ids_lens;
};

}  // namespace llm
