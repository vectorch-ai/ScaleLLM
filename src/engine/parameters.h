#pragma once

#include <torch/torch.h>

#include "models/parameters.h"
#include "sampling/parameters.h"

namespace llm {

// input for the model that encapsulates all the necessary
// input information.
struct ModelInput {
  // flatten token ids
  torch::Tensor token_ids;
  // flatten positions
  torch::Tensor positions;
  // input parameters, mainly for attention
  InputParameters input_params;
  // sampling parameters, mainly for sampling
  SamplingParameters sampling_params;
};

// output for the model that encapsulates all the necessary
// output information.
struct ModelOutput {
  // sample carried over from input, used for speculative decoding
  std::vector<bool> do_sample;

  // output of sampling
  SampleOutput sample_output;

  // logits for selected indices
  torch::Tensor logits;

  // torch::Tensor logprob;
};

}  // namespace llm