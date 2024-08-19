#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include "linear.h"
#include "model_loader/state_dict.h"
#include "model_parallel/parallel_args.h"
#include "quantization/quant_args.h"

namespace llm {

class FusedColumnParallelLinearImpl : public torch::nn::Module {
 public:
  FusedColumnParallelLinearImpl(int64_t in_features,
                                const std::vector<int64_t>& out_features,
                                bool bias,
                                bool gather_output,
                                const QuantArgs& quant_args,
                                const ParallelArgs& parallel_args,
                                const torch::TensorOptions& options);

  std::vector<torch::Tensor> forward(torch::Tensor input);

  // load_state_dict for fused weights
  void load_state_dict(const StateDict& state_dict,
                       const std::vector<std::string>& prefixes);

  void verify_loaded_weights(const std::string& prefix = "") const;

  // whether the linear layer is fused
  bool fused() const { return fused_; }

 private:
  // non-fused linear layers
  std::vector<ColumnParallelLinear> parallel_linears_;

  // fused linear layer
  ColumnParallelLinear fused_linear_{nullptr};

  // sizes for each split
  std::vector<int64_t> split_sizes_;

  // whether the linear layer is fused
  bool fused_ = false;
};
TORCH_MODULE(FusedColumnParallelLinear);

}  // namespace llm
