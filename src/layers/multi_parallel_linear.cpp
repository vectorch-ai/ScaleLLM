#include "multi_parallel_linear.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include "model_parallel/parallel_args.h"
#include "parallel_linear.h"
#include "quantization/quant_args.h"

namespace llm {

MultiColumnParallelLinearImpl::MultiColumnParallelLinearImpl(
    int64_t in_features,
    const std::vector<int64_t>& out_features_vec,
    const std::vector<std::string>& prefixes,
    bool bias,
    bool gather_output,
    const QuantArgs& quant_args,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options) {
  // check if the linear layers can be fused
  fused_ = quant_args.can_be_fused();
  if (fused_) {
    // fused linear layer
    fused_linear_ = register_module("fused_linear",
                                    FusedColumnParallelLinear(in_features,
                                                              out_features_vec,
                                                              prefixes,
                                                              bias,
                                                              gather_output,
                                                              parallel_args,
                                                              options),
                                    /*selector=*/nullptr);
  } else {
    // non-fused linear layers
    grouped_linear_ =
        register_module("grouped_linear",
                        GroupedColumnParallelLinear(in_features,
                                                    out_features_vec,
                                                    prefixes,
                                                    bias,
                                                    gather_output,
                                                    parallel_args,
                                                    options),
                        /*selector=*/nullptr);
  }
}

std::vector<torch::Tensor> MultiColumnParallelLinearImpl::forward(
    torch::Tensor input) {
  if (fused_) {
    return fused_linear_(input);
  }
  return grouped_linear_(input);
}
}  // namespace llm
