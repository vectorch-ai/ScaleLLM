#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include "model_loader/state_dict.h"
#include "model_parallel.h"
#include "models/parallel_args.h"

namespace llm {

// quantized linear layers

// Linear layer with column parallelism.
// The linear layer is defined as Y = XA + b. A is parallelized along
// its second dimension as A = [A_1, ..., A_p].
class QuantColumnParallelLinearImpl : public torch::nn::Module {
 public:
};
TORCH_MODULE(QuantColumnParallelLinear);

}  // namespace llm
