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
#include "quantization/qlinear_exllama_impl.h"
#include "quantization/qlinear_gptq_impl.h"

DEFINE_string(
    qlinear_gptq_impl,
    "auto",
    "type of qlinear gptq impl, slow, cuda, exllama, exllamav2 or auto");

namespace llm {
namespace {
std::shared_ptr<ParallelLinearImpl> create_column_parallel_qlinear_by_impl(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    bool gather_output,
    const QuantizationArgs& quant_args,
    const ParallelArgs& parallel_args,
    torch::ScalarType dtype,
    const torch::Device& device) {
  if (boost::iequals(FLAGS_qlinear_gptq_impl, "slow")) {
    return std::make_shared<ColumnParallelQLinearImpl>(in_features,
                                                       out_features,
                                                       bias,
                                                       quant_args,
                                                       /*qweight_pack_dim=*/0,
                                                       gather_output,
                                                       parallel_args,
                                                       dtype,
                                                       device);
  }
  if (boost::iequals(FLAGS_qlinear_gptq_impl, "cuda")) {
    return std::make_shared<ColumnParallelQLinearGPTQImpl>(in_features,
                                                           out_features,
                                                           bias,
                                                           quant_args,
                                                           gather_output,
                                                           parallel_args,
                                                           dtype,
                                                           device);
  }
  if (boost::iequals(FLAGS_qlinear_gptq_impl, "exllama")) {
    return std::make_shared<ColumnParallelQLinearExllamaImpl>(in_features,
                                                              out_features,
                                                              bias,
                                                              quant_args,
                                                              gather_output,
                                                              parallel_args,
                                                              dtype,
                                                              device);
  }
  return nullptr;
}

std::shared_ptr<ParallelLinearImpl> create_row_parallel_qlinear_by_impl(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    bool input_is_parallelized,
    const QuantizationArgs& quant_args,
    const ParallelArgs& parallel_args,
    torch::ScalarType dtype,
    const torch::Device& device) {
  if (boost::iequals(FLAGS_qlinear_gptq_impl, "slow")) {
    return std::make_shared<RowParallelQLinearImpl>(in_features,
                                                    out_features,
                                                    bias,
                                                    quant_args,
                                                    /*qweight_pack_dim=*/0,
                                                    input_is_parallelized,
                                                    parallel_args,
                                                    dtype,
                                                    device);
  }
  if (boost::iequals(FLAGS_qlinear_gptq_impl, "cuda")) {
    return std::make_shared<RowParallelQLinearGPTQImpl>(in_features,
                                                        out_features,
                                                        bias,
                                                        quant_args,
                                                        input_is_parallelized,
                                                        parallel_args,
                                                        dtype,
                                                        device);
  }
  if (boost::iequals(FLAGS_qlinear_gptq_impl, "exllama")) {
    return std::make_shared<RowParallelQLinearExllamaImpl>(
        in_features,
        out_features,
        bias,
        quant_args,
        input_is_parallelized,
        parallel_args,
        dtype,
        device);
  }
  return nullptr;
}

std::shared_ptr<ParallelLinearImpl> create_column_parallel_qlinear(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    bool gather_output,
    const QuantizationArgs& quant_args,
    const ParallelArgs& parallel_args,
    torch::ScalarType dtype,
    const torch::Device& device) {
  if (auto qlinear = create_column_parallel_qlinear_by_impl(in_features,
                                                            out_features,
                                                            bias,
                                                            gather_output,
                                                            quant_args,
                                                            parallel_args,
                                                            dtype,
                                                            device)) {
    return qlinear;
  }
  if (boost::iequals(quant_args.quant_method(), "gptq")) {
    // use exllama for 4 bits which is faster
    if (quant_args.bits() == 4) {
      return std::make_shared<ColumnParallelQLinearExllamaImpl>(in_features,
                                                                out_features,
                                                                bias,
                                                                quant_args,
                                                                gather_output,
                                                                parallel_args,
                                                                dtype,
                                                                device);
    }
    return std::make_shared<ColumnParallelQLinearGPTQImpl>(in_features,
                                                           out_features,
                                                           bias,
                                                           quant_args,
                                                           gather_output,
                                                           parallel_args,
                                                           dtype,
                                                           device);
  }
  if (boost::iequals(quant_args.quant_method(), "awq") ||
      boost::iequals(quant_args.quant_method(), "GEMM")) {
    // default to use awq implementation for gemm
    return std::make_shared<ColumnParallelQLinearAWQImpl>(in_features,
                                                          out_features,
                                                          bias,
                                                          quant_args,
                                                          gather_output,
                                                          parallel_args,
                                                          dtype,
                                                          device);
  }
  // not supported quant method
  LOG(FATAL) << "Unsupported quant method: " << quant_args.quant_method();
}

std::shared_ptr<ParallelLinearImpl> create_row_parallel_qlinear(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    bool input_is_parallelized,
    const QuantizationArgs& quant_args,
    const ParallelArgs& parallel_args,
    torch::ScalarType dtype,
    const torch::Device& device) {
  if (auto qlinear = create_row_parallel_qlinear_by_impl(in_features,
                                                         out_features,
                                                         bias,
                                                         input_is_parallelized,
                                                         quant_args,
                                                         parallel_args,
                                                         dtype,
                                                         device)) {
    return qlinear;
  }
  if (boost::iequals(quant_args.quant_method(), "gptq")) {
    // use exllama for 4 bits which is faster
    if (quant_args.bits() == 4) {
      // TODO: double chekc if exllama supports row tensor parallelism with
      // act-order.
      return std::make_shared<RowParallelQLinearExllamaImpl>(
          in_features,
          out_features,
          bias,
          quant_args,
          input_is_parallelized,
          parallel_args,
          dtype,
          device);
    }
    return std::make_shared<RowParallelQLinearGPTQImpl>(in_features,
                                                        out_features,
                                                        bias,
                                                        quant_args,
                                                        input_is_parallelized,
                                                        parallel_args,
                                                        dtype,
                                                        device);
  }
  if (boost::iequals(quant_args.quant_method(), "awq") ||
      boost::iequals(quant_args.quant_method(), "GEMM")) {
    // default to use awq implementation for gemm
    return std::make_shared<RowParallelQLinearAWQImpl>(in_features,
                                                       out_features,
                                                       bias,
                                                       quant_args,
                                                       input_is_parallelized,
                                                       parallel_args,
                                                       dtype,
                                                       device);
  }
  // not supported quant method
  LOG(FATAL) << "Unsupported quant method: " << quant_args.quant_method();
}

std::shared_ptr<ParallelLinearImpl> create_column_parallel_linear(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    bool gather_output,
    const QuantizationArgs& quant_args,
    const ParallelArgs& parallel_args,
    torch::ScalarType dtype,
    const torch::Device& device) {
  if (!quant_args.quant_method().empty()) {
    return create_column_parallel_qlinear(in_features,
                                          out_features,
                                          bias,
                                          gather_output,
                                          quant_args,
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
  if (!quant_args.quant_method().empty()) {
    return create_row_parallel_qlinear(in_features,
                                       out_features,
                                       bias,
                                       input_is_parallelized,
                                       quant_args,
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
