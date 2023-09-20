#include "linear.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <memory>

#include "linear_impl.h"
#include "model_loader/state_dict.h"
#include "models/parallel_args.h"
#include "qlinear_impl.h"

namespace llm {
namespace {
std::shared_ptr<ParallelLinearImpl> create_column_parallel_linear(
    int64_t in_features,
    int64_t out_features,
    bool gather_output,
    const ParallelArgs& parallel_args,
    const torch::ScalarType& dtype,
    const torch::Device& device) {
  return std::make_shared<ColumnParallelLinearImpl>(
      in_features, out_features, gather_output, parallel_args, dtype, device);
}

std::shared_ptr<ParallelLinearImpl> create_row_parallel_linear(
    int64_t in_features,
    int64_t out_features,
    bool input_is_parallelized,
    const ParallelArgs& parallel_args,
    const torch::ScalarType& dtype,
    const torch::Device& device) {
  return std::make_shared<RowParallelLinearImpl>(in_features,
                                                 out_features,
                                                 input_is_parallelized,
                                                 parallel_args,
                                                 dtype,
                                                 device);
}
}  // namespace

// construct a ColumnParallelLinear.
// chose right implementation based on the args.
ColumnParallelLinear::ColumnParallelLinear(int64_t in_features,
                                           int64_t out_features,
                                           bool gather_output,
                                           const ParallelArgs& parallel_args,
                                           const torch::ScalarType& dtype,
                                           const torch::Device& device)
    : ModuleHolder(create_column_parallel_linear(in_features,
                                                 out_features,
                                                 gather_output,
                                                 parallel_args,
                                                 dtype,
                                                 device)) {}

// construct a rotary positional embedding.
// chose right implementation based on the args.
RowParallelLinear::RowParallelLinear(int64_t in_features,
                                     int64_t out_features,
                                     bool input_is_parallelized,
                                     const ParallelArgs& parallel_args,
                                     const torch::ScalarType& dtype,
                                     const torch::Device& device)
    : ModuleHolder(create_row_parallel_linear(in_features,
                                              out_features,
                                              input_is_parallelized,
                                              parallel_args,
                                              dtype,
                                              device)) {}
}  // namespace llm
