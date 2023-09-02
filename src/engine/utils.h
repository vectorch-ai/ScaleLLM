#pragma once

#include <torch/torch.h>

#include <vector>

#include "models/input_parameters.h"
#include "request/request.h"

namespace llm {

class Utils {
 public:
  static void prepare_inputs(const std::vector<Request*>& batch,
                             torch::Tensor* input_token_ids,
                             torch::Tensor* input_positions,
                             InputParameters* input_params,
                             SamplingParameters* sampling_params);
};

}  // namespace llm
