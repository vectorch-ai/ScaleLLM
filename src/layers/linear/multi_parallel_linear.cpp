#include "multi_parallel_linear.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include "layers/quantization/quant_args.h"
#include "model_parallel/model_parallel.h"
#include "model_parallel/parallel_args.h"
#include "parallel_linear.h"

namespace llm {

FusedColumnParallelLinearImpl::FusedColumnParallelLinearImpl(
    int64_t in_features,
    const std::vector<int64_t>& out_features_vec,
    const std::vector<std::string>& prefixes,
    bool bias,
    bool gather_output,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options)
    : gather_output_(gather_output), parallel_args_(parallel_args) {
  const auto rank = parallel_args_.rank();
  const auto world_size = parallel_args_.world_size();

  // calculate split size for each prefix
  split_sizes_.reserve(out_features_vec.size());
  for (const auto& out_features : out_features_vec) {
    CHECK(out_features % world_size == 0)
        << "out_features " << out_features << " not divisible by world_size "
        << world_size;
    split_sizes_.push_back(out_features / world_size);
  }

  const int64_t fused_out_features =
      std::accumulate(split_sizes_.begin(), split_sizes_.end(), int64_t(0));

  // allocate fused weight
  weight_ = torch::empty({fused_out_features, in_features}, options);
  const auto weights = weight_.split(split_sizes_, /*dim=*/0);
  // register sharded weights for each prefix
  for (size_t i = 0; i < prefixes.size(); ++i) {
    const auto& prefix = prefixes[i];
    const auto& weight = weights[i];
    // register the weight as a parameter to make sure it is moved to the
    register_sharded_parameter(detail::join_name(prefix, "weight"),
                               /*dim=*/0,
                               rank,
                               world_size,
                               weight);
  }

  if (bias) {
    bias_ = torch::empty({fused_out_features}, options);
    const auto biases = bias_.split(split_sizes_, /*dim=*/0);

    // register sharded weights for each prefix
    for (size_t i = 0; i < prefixes.size(); ++i) {
      const auto& prefix = prefixes[i];
      const auto& bias = biases[i];
      register_sharded_parameter(detail::join_name(prefix, "bias"),
                                 /*dim=*/0,
                                 rank,
                                 world_size,
                                 bias);
    }
  }
}

std::vector<torch::Tensor> FusedColumnParallelLinearImpl::forward(
    torch::Tensor input) {
  namespace F = torch::nn::functional;
  auto output = F::linear(input, weight_, bias_);
  if (parallel_args_.world_size() > 1 && gather_output_) {
    output = gather_from_model_parallel_region(output, parallel_args_);
  }
  return output.split(split_sizes_, /*dim=*/1);
}

GroupedColumnParallelLinearImpl::GroupedColumnParallelLinearImpl(
    int64_t in_features,
    const std::vector<int64_t>& out_features_vec,
    const std::vector<std::string>& prefixes,
    bool bias,
    bool gather_output,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options) {
  // register linear layers one by one
  parallel_linears_.reserve(out_features_vec.size());
  for (size_t i = 0; i < out_features_vec.size(); ++i) {
    const auto& prefix = prefixes[i];
    const auto out_features = out_features_vec[i];
    const auto linear = register_module(
        "linear_" + std::to_string(i),
        std::make_shared<ColumnParallelLinearImpl>(in_features,
                                                   out_features,
                                                   bias,
                                                   gather_output,
                                                   parallel_args,
                                                   options,
                                                   prefix),
        /*selector=*/nullptr);

    parallel_linears_.emplace_back(linear);
  }
}

std::vector<torch::Tensor> GroupedColumnParallelLinearImpl::forward(
    torch::Tensor input) {
  std::vector<torch::Tensor> outputs;
  outputs.reserve(parallel_linears_.size());
  for (auto& parallel_linear : parallel_linears_) {
    outputs.push_back(parallel_linear->forward(input));
  }
  return outputs;
}

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
  std::shared_ptr<MultiParallelLinearImpl> linear;
  if (quant_args.can_be_fused()) {
    // fused linear layer
    linear = register_module("fused_linear",
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
    linear = register_module("grouped_linear",
                             GroupedColumnParallelLinear(in_features,
                                                         out_features_vec,
                                                         prefixes,
                                                         bias,
                                                         gather_output,
                                                         parallel_args,
                                                         options),
                             /*selector=*/nullptr);
  }
  linear_ = linear;
}

std::vector<torch::Tensor> MultiColumnParallelLinearImpl::forward(
    torch::Tensor input) {
  return linear_(input);
}
}  // namespace llm
