#pragma once
#include <torch/torch.h>
#include <cstdint>
#include <vector>

namespace llm {

// SamplingParameter is used to specify sampling parameters for a
// request/sequence.
struct SamplingParameter {
  float frequency_penalty = 0.0;
  float presence_penalty = 0.0;
  float repetition_penalty = 1.0;
  float temperature = 1.0;
  float top_p = 1.0;
  int64_t top_k = 0;
  bool do_sample = false;
  uint64_t seed = 0;
};

// SamplingParameters is used to specify sampling parameters for a batch of
// requests/sequences.
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

struct SampleOutput {
  // [num_seq] LongTensor
  torch::Tensor next_tokens;
  
  // [num_seq] FloatTensor
  // torch::Tensor next_logprobs;
};

}  // namespace llm
