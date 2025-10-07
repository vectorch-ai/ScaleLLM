#include "parallel_linear.h"

#include <c10/core/TensorImpl.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include "model_parallel/model_parallel.h"
#include "module/module.h"

namespace llm {

// Linear layer with column parallelism.
ColumnParallelLinearImpl::ColumnParallelLinearImpl(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    bool gather_output,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options,
    const std::string& prefix)
    : gather_output_(gather_output), parallel_args_(parallel_args) {
  const auto rank = parallel_args_.rank();
  const auto world_size = parallel_args_.world_size();
  CHECK(out_features % world_size == 0)
      << "out_features " << out_features << " not divisible by world_size "
      << world_size;
  const int64_t out_features_per_partition = out_features / world_size;

  // Note: torch.nn.functional.linear performs XA^T + b and as a result
  // we allocate the transpose.
  weight_ = register_sharded_parameter(
      detail::join_name(prefix, "weight"),
      /*dim=*/0,
      rank,
      world_size,
      torch::empty({out_features_per_partition, in_features}, options));

  if (bias) {
    bias_ = register_sharded_parameter(
        detail::join_name(prefix, "bias"),
        /*dim=*/0,
        rank,
        world_size,
        torch::empty({out_features_per_partition}, options));
  }
}

torch::Tensor ColumnParallelLinearImpl::forward(torch::Tensor input) {
  namespace F = torch::nn::functional;
  auto output = F::linear(input, weight_, bias_);
  if (parallel_args_.world_size() > 1 && gather_output_) {
    output = gather_from_model_parallel_region(output, parallel_args_);
  }
  return output;
}

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

// Linear layer with row parallelism.
RowParallelLinearImpl::RowParallelLinearImpl(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    bool input_is_parallelized,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options)
    : input_is_parallelized_(input_is_parallelized),
      parallel_args_(parallel_args) {
  const auto rank = parallel_args_.rank();
  const auto world_size = parallel_args_.world_size();
  CHECK(in_features % world_size == 0)
      << "in_features " << in_features << " not divisible by world_size "
      << world_size;
  const int64_t in_features_per_partition = in_features / world_size;
  // Allocate the transpose since linear performs XA^T.
  weight_ = register_sharded_parameter(
      "weight",
      /*dim=*/1,
      rank,
      world_size,
      torch::empty({out_features, in_features_per_partition}, options));

  if (bias) {
    bias_ = register_parameter("bias", torch::empty({out_features}, options));
  }
}

torch::Tensor RowParallelLinearImpl::forward(torch::Tensor input) {
  namespace F = torch::nn::functional;
  if (!input_is_parallelized_) {
    input = scatter_to_model_parallel_region(input, parallel_args_);
  }
  auto output = F::linear(input, weight_);
  if (parallel_args_.world_size() > 1) {
    output = reduce_from_model_parallel_region(output, parallel_args_);
  }
  // N.B. need to apply bias after the reduce
  if (bias_.defined()) {
    output.add_(bias_);
  }
  return output;
}

}  // namespace llm
