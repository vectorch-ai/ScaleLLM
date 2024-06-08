#pragma once
#include <torch/torch.h>

#include <cstdint>
#include <vector>

#include "common/tensor_helper.h"

namespace llm {

// SamplingParameter is used to specify sampling parameters for a
// request/sequence.
struct SamplingParameter {
  // ########### following parameters are used for logit processing ########
  float frequency_penalty = 0.0;
  float presence_penalty = 0.0;
  float repetition_penalty = 1.0;
  float temperature = 0.7;
  float top_p = 1.0;
  int64_t top_k = -1;
  bool logprobs = false;
  int64_t top_logprobs = 0;

  // ############### following parameters are used for sampling ###############
  bool do_sample = false;

  // not used for now
  uint64_t seed = 0;
};

// SamplingParameters is used to specify sampling parameters for a batch of
// requests/sequences.
struct SamplingParameters {
  // initialize the sampling parameters from the given sampling parameters
  void init(const std::vector<const SamplingParameter*>& sampling_params,
            const std::vector<int32_t>& selected_token_idxes,
            const std::vector<int32_t>& sample_idxes,
            const std::vector<std::vector<int64_t>>& unique_token_ids_vec,
            const std::vector<std::vector<int32_t>>& unique_token_counts_vec,
            const std::vector<int32_t>& unique_token_lens_vec);

  SamplingParameters to(const torch::Device& device,
                        torch::ScalarType dtype) const {
    SamplingParameters params;

    // all tensors should be on the same device
    params.selected_token_idxes = safe_to(selected_token_idxes, device);

    auto options = torch::device(device).dtype(dtype);
    params.frequency_penalties = safe_to(frequency_penalties, options);
    params.presence_penalties = safe_to(presence_penalties, options);
    params.repetition_penalties = safe_to(repetition_penalties, options);
    params.temperatures = safe_to(temperatures, options);
    params.top_p = safe_to(top_p, options);
    params.top_k = safe_to(top_k, device);

    params.unique_token_ids = safe_to(unique_token_ids, device);
    params.unique_token_counts = safe_to(unique_token_counts, device);
    params.unique_token_ids_lens = safe_to(unique_token_ids_lens, device);

    params.sample_idxes = safe_to(sample_idxes, device);
    params.do_sample = safe_to(do_sample, device);
    params.logprobs = logprobs;
    params.top_logprobs = top_logprobs;

    return params;
  }

  // ########### following parameters are used for logit processing ########
  // selected tokens are tokens for sampling the next token,
  // including the generated tokens and the last prompt token
  // IntTensor
  torch::Tensor selected_token_idxes;

  // [num_tokens] FloatTensor
  torch::Tensor frequency_penalties;

  // [num_tokens] FloatTensor
  torch::Tensor presence_penalties;

  // [num_tokens] FloatTensor
  torch::Tensor repetition_penalties;

  // [num_tokens] FloatTensor
  torch::Tensor temperatures;

  // [num_tokens] FloatTensor
  torch::Tensor top_p;

  // [num_tokens] LongTensor
  torch::Tensor top_k;

  // the unique token id and count of each sequence in the batch.
  // [num_tokens, max_unique_tokens] LongTensor
  torch::Tensor unique_token_ids;

  // [num_tokens, max_unique_tokens] IntTensor
  torch::Tensor unique_token_counts;

  // the number of unique tokens in each sequence.
  // [num_tokens] IntTensor
  torch::Tensor unique_token_ids_lens;

  // ############### following parameters are used for sampling ###############
  // the last index of the selected tokens for sampling.
  // [num_seqs] IntTensor
  torch::Tensor sample_idxes;

  // whether to sample for each sequence.
  // [num_seqs] BoolTensor
  torch::Tensor do_sample;

  // whether to output logprobs for each generated token.
  bool logprobs = false;

  // the number of top logprobs to output for each generated token.
  // only used when logprobs is true.
  int64_t top_logprobs = 0;
};

struct SampleOutput {
  // [num_seq] LongTensor
  torch::Tensor next_tokens;

  // [num_seq] FloatTensor
  torch::Tensor probs;

  // [num_seq] FloatTensor
  torch::Tensor logprobs;

  // [num_seq, top_k] FloatTensor
  torch::Tensor top_logprobs;
  // [num_seq, top_k] LongTensor
  torch::Tensor top_tokens;
};

}  // namespace llm
