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
// output information. The output parameters should be as small as possible
// to avoid transferring large tensors between host and device.
struct ModelOutput {
  // [num_seq] LongTensor
  torch::Tensor next_tokens;

  // [num_seq]
  // torch::Tensor next_logprob;
};

}  // namespace llm