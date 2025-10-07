#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

// #include "linear.h"
#include "model_parallel/parallel_args.h"
#include "module/module.h"
#include "module/module_holder.h"
#include "parallel_linear.h"
#include "quantization/quant_args.h"

namespace llm {

class MultiColumnParallelLinearImpl : public Module {
 public:
  MultiColumnParallelLinearImpl(int64_t in_features,
                                const std::vector<int64_t>& out_features,
                                const std::vector<std::string>& prefixes,
                                bool bias,
                                bool gather_output,
                                const QuantArgs& quant_args,
                                const ParallelArgs& parallel_args,
                                const torch::TensorOptions& options);

  std::vector<torch::Tensor> forward(torch::Tensor input);

  // whether the linear layer is fused
  bool fused() const { return fused_; }

 private:
  // non-fused linear layers
  GroupedColumnParallelLinear grouped_linear_{nullptr};

  // fused linear layer
  FusedColumnParallelLinear fused_linear_{nullptr};

  // whether the linear layer is fused
  bool fused_ = false;
};
LLM_MODULE(MultiColumnParallelLinear);

}  // namespace llm
