#include "linear.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <memory>

#include "linear_impl.h"
#include "model_loader/state_dict.h"
#include "models/args.h"
#include "quantization/qlinear_awq_impl.h"
#include "quantization/qlinear_gptq_impl.h"

namespace llm {
namespace {
std::shared_ptr<ParallelLinearImpl> create_column_parallel_linear(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    bool gather_output,
    const QuantizationArgs& quant_args,
    const ParallelArgs& parallel_args,
    torch::ScalarType dtype,
    const torch::Device& device) {
  if (boost::iequals(quant_args.quant_method(), "gptq")) {
    return std::make_shared<ColumnParallelQLinearGPTQImpl>(
        in_features,
        out_features,
        bias,
        quant_args.bits(),
        quant_args.group_size(),
        gather_output,
        parallel_args,
        dtype,
        device);
  }
  if (boost::iequals(quant_args.quant_method(), "GEMM")) {
    return std::make_shared<ColumnParallelQLinearAWQImpl>(
        in_features,
        out_features,
        bias,
        quant_args.bits(),
        quant_args.group_size(),
        gather_output,
        parallel_args,
        dtype,
        device);
  }
  return std::make_shared<ColumnParallelLinearImpl>(in_features,
                                                    out_features,
                                                    bias,
                                                    gather_output,
                                                    parallel_args,
                                                    dtype,
                                                    device);
}

std::shared_ptr<ParallelLinearImpl> create_row_parallel_linear(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    bool input_is_parallelized,
    const QuantizationArgs& quant_args,
    const ParallelArgs& parallel_args,
    torch::ScalarType dtype,
    const torch::Device& device) {
  if (boost::iequals(quant_args.quant_method(), "gptq")) {
    return std::make_shared<RowParallelQLinearGPTQImpl>(in_features,
                                                        out_features,
                                                        bias,
                                                        quant_args.bits(),
                                                        quant_args.group_size(),
                                                        input_is_parallelized,
                                                        parallel_args,
                                                        dtype,
                                                        device);
  }
  if (boost::iequals(quant_args.quant_method(), "GEMM")) {
    return std::make_shared<RowParallelQLinearAWQImpl>(in_features,
                                                       out_features,
                                                       bias,
                                                       quant_args.bits(),
                                                       quant_args.group_size(),
                                                       input_is_parallelized,
                                                       parallel_args,
                                                       dtype,
                                                       device);
  }
  return std::make_shared<RowParallelLinearImpl>(in_features,
                                                 out_features,
                                                 bias,
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
                                           bool bias,
                                           bool gather_output,
                                           const QuantizationArgs& quant_args,
                                           const ParallelArgs& parallel_args,
                                           torch::ScalarType dtype,
                                           const torch::Device& device)
    : ModuleHolder(create_column_parallel_linear(in_features,
                                                 out_features,
                                                 bias,
                                                 gather_output,
                                                 quant_args,
                                                 parallel_args,
                                                 dtype,
                                                 device)) {}

ColumnParallelLinear::ColumnParallelLinear(int64_t in_features,
                                           int64_t out_features,
                                           bool bias,
                                           bool gather_output,
                                           const ParallelArgs& parallel_args,
                                           torch::ScalarType dtype,
                                           const torch::Device& device)
    : ModuleHolder(create_column_parallel_linear(in_features,
                                                 out_features,
                                                 bias,
                                                 gather_output,
                                                 {}, /*quant_args*/
                                                 parallel_args,
                                                 dtype,
                                                 device)) {}

// construct a rotary positional embedding.
// chose right implementation based on the args.
RowParallelLinear::RowParallelLinear(int64_t in_features,
                                     int64_t out_features,
                                     bool bias,
                                     bool input_is_parallelized,
                                     const QuantizationArgs& quant_args,
                                     const ParallelArgs& parallel_args,
                                     torch::ScalarType dtype,
                                     const torch::Device& device)
    : ModuleHolder(create_row_parallel_linear(in_features,
                                              out_features,
                                              bias,
                                              input_is_parallelized,
                                              quant_args,
                                              parallel_args,
                                              dtype,
                                              device)) {}
}  // namespace llm
