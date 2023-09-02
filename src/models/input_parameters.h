#pragma once

#include <torch/torch.h>
#include <request/request.h>

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
    params.sample_idx = sample_idx.to(device);
    params.token_ids = token_ids.to(device);
    return params;
  }

  // total number of tokens in prompt sequences.
  int64_t num_prompt_tokens = 0;

  // cumulative sequence length of each sequence.
  // used in prefill stage to determine the token range for each sequence
  // [num_prompt_seq + 1]
  // for example: 3 sequences with length 2, 3, 4,
  // the cu_seq_lens is [0, 2, 5, 9]
  torch::Tensor cu_seq_lens;

  // maximum sequence length for prompt sequences.
  int32_t max_seq_len = 0;

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

  // the index of the last token of each sequence in the tokens.
  // for prompt sequence, it is the index of last token in the prompt.
  // for decode sequence, it is the index of the token. (only one token)
  torch::Tensor sample_idx;

  // the token ids of the prompt.
  // used in logit processor to calculate frequency of each token
  // [num_seq (num_prompt_seq + num_decode_seq), max_tokens_len]
  torch::Tensor token_ids;
};

struct SamplingParameters {
  void add(const SamplingParameter& p) {
    frequency_penalties.push_back(p.frequency_penalty);
    presence_penalties.push_back(p.presence_penalty);
    repetition_penalties.push_back(p.repetition_penalty);
    temperatures.push_back(p.temperature);
    top_p.push_back(p.top_p);
    top_k.push_back(p.top_k);
    seeds.push_back(p.seed);

    // need to do sample if any of following is true
    // * do_sample = true (from user input)
    // * temperature != 0.0
    // * top_p != 1.0 (default)
    // * top_k != 0 (default)
    const bool sample =
        p.do_sample || p.temperature != 0.0 || p.top_p != 1.0 || p.top_k != 0;
    do_sample.push_back(sample);
  }

  // following are used for sampling, with shape [num_request]
  // default = 0.0
  std::vector<float> frequency_penalties;

  // default = 0.0
  std::vector<float> presence_penalties;

  // default = 1.0
  std::vector<float> repetition_penalties;

  // default = 0.0
  std::vector<float> temperatures;

  // default = 1.0
  std::vector<float> top_p;

  // default = 0
  std::vector<int64_t> top_k;

  // need to be set to true if any of following is true
  // * do_sample = true (from user input)
  // * temperature != 0.0
  // * top_p != 1.0 (default)
  // * top_k != 0 (default)
  std::vector<bool> do_sample;

  // default = 0, use global generator
  std::vector<uint64_t> seeds;
};

// output parameters for the model that encapsulates all the necessary
// output information. The output parameters should be as small as possible
// to avoid transferring large tensors between host and device.
struct OutputParameters {
  // [num_seq]
  torch::Tensor next_tokens;

  // [num_seq]
  torch::Tensor next_logprob;
};

}  // namespace llm
