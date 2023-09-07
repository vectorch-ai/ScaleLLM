#pragma once

#include <torch/torch.h>

#include <vector>

#include "models/parameters.h"
#include "request/sequence.h"

namespace llm {

class Utils {
 public:
  static void prepare_inputs(const std::vector<Sequence*>& batch,
                             int32_t block_size,
                             torch::Tensor* input_token_ids,
                             torch::Tensor* input_positions,
                             torch::Tensor* seq_indices,
                             InputParameters* input_params,
                             SamplingParameters* sampling_params);
};

}  // namespace llm
