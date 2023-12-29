#pragma once

#include <torch/torch.h>

#include <vector>

#include "models/input_parameters.h"
#include "request/sequence.h"

namespace llm {

class Utils {
 public:
  static void prepare_inputs(const std::vector<Sequence*>& batch,
                             int32_t block_size,
                             torch::Tensor* flatten_token_ids,
                             torch::Tensor* flatten_positions,
                             torch::Tensor* seq_idxes,
                             InputParameters* input_params,
                             SamplingParameters* sampling_params);
  static void prepare_validate_inputs(const std::vector<Sequence*>& batch,
                                      int32_t block_size,
                                      torch::Tensor* flatten_token_ids,
                                      torch::Tensor* flatten_positions,
                                      torch::Tensor* seq_idxes,
                                      InputParameters* input_params,
                                      SamplingParameters* sampling_params);
};

}  // namespace llm
