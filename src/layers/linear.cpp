#include "linear.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <boost/algorithm/string.hpp>
#include <memory>

#include "linear_impl.h"
#include "quantization/qlinear_awq_impl.h"
#include "quantization/qlinear_awq_marlin_impl.h"
#include "quantization/qlinear_exllamav2_impl.h"
#include "quantization/qlinear_gptq_impl.h"
#include "quantization/qlinear_gptq_marlin_impl.h"

DEFINE_string(
    qlinear_gptq_impl,
    "auto",
    "type of qlinear gptq impl: slow, cuda, exllamav2, marlin or auto");

namespace llm {
namespace {
#define MAKE_ROW_PARALLEL_QLINEAR(QLinearlImplClass)         \
  std::make_shared<QLinearlImplClass>(in_features,           \
                                      out_features,          \
                                      bias,                  \
                                      quant_args,            \
                                      input_is_parallelized, \
                                      parallel_args,         \
                                      options);

#define MAKE_COLUMN_PARALLEL_QLINEAR(QLinearlImplClass) \
  std::make_shared<QLinearlImplClass>(in_features,      \
                                      out_features,     \
                                      bias,             \
                                      quant_args,       \
                                      gather_output,    \
                                      parallel_args,    \
                                      options);

#define MAKE_ROW_PARALLEL_LINEAR(LinearlImplClass)          \
  std::make_shared<LinearlImplClass>(in_features,           \
                                     out_features,          \
                                     bias,                  \
                                     input_is_parallelized, \
                                     parallel_args,         \
                                     options);

#define MAKE_COLUMN_PARALLEL_LINEAR(LinearlImplClass) \
  std::make_shared<LinearlImplClass>(                 \
      in_features, out_features, bias, gather_output, parallel_args, options);

std::shared_ptr<ParallelLinearImpl> create_column_parallel_qlinear_by_impl(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    bool gather_output,
    const QuantArgs& quant_args,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options) {
  if (boost::iequals(FLAGS_qlinear_gptq_impl, "slow")) {
    return std::make_shared<ColumnParallelQLinearImpl>(in_features,
                                                       out_features,
                                                       bias,
                                                       quant_args,
                                                       /*qweight_pack_dim=*/0,
                                                       gather_output,
                                                       parallel_args,
                                                       options);
  }
  if (boost::iequals(FLAGS_qlinear_gptq_impl, "cuda")) {
    return MAKE_COLUMN_PARALLEL_QLINEAR(ColumnParallelQLinearGPTQImpl);
  }
  if (boost::iequals(FLAGS_qlinear_gptq_impl, "exllamav2")) {
    return MAKE_COLUMN_PARALLEL_QLINEAR(ColumnParallelQLinearExllamav2Impl);
  }
  if (boost::iequals(FLAGS_qlinear_gptq_impl, "marlin")) {
    return MAKE_COLUMN_PARALLEL_QLINEAR(ColumnParallelQLinearGPTQMarlinImpl);
  }
  return nullptr;
}

std::shared_ptr<ParallelLinearImpl> create_row_parallel_qlinear_by_impl(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    bool input_is_parallelized,
    const QuantArgs& quant_args,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options) {
  if (boost::iequals(FLAGS_qlinear_gptq_impl, "slow")) {
    return std::make_shared<RowParallelQLinearImpl>(in_features,
                                                    out_features,
                                                    bias,
                                                    quant_args,
                                                    /*qweight_pack_dim=*/0,
                                                    input_is_parallelized,
                                                    parallel_args,
                                                    options);
  }
  if (boost::iequals(FLAGS_qlinear_gptq_impl, "cuda")) {
    return MAKE_ROW_PARALLEL_QLINEAR(RowParallelQLinearGPTQImpl);
  }
  if (boost::iequals(FLAGS_qlinear_gptq_impl, "exllamav2")) {
    return MAKE_ROW_PARALLEL_QLINEAR(RowParallelQLinearExllamav2Impl);
  }
  if (boost::iequals(FLAGS_qlinear_gptq_impl, "marlin")) {
    return MAKE_ROW_PARALLEL_QLINEAR(RowParallelQLinearGPTQMarlinImpl);
  }
  return nullptr;
}

std::shared_ptr<ParallelLinearImpl> create_column_parallel_qlinear(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    bool gather_output,
    const QuantArgs& quant_args,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options) {
  if (auto qlinear = create_column_parallel_qlinear_by_impl(in_features,
                                                            out_features,
                                                            bias,
                                                            gather_output,
                                                            quant_args,
                                                            parallel_args,
                                                            options)) {
    return qlinear;
  }
  if (boost::iequals(quant_args.quant_method(), "gptq")) {
    // default to use marlin implementation for gptq
    return MAKE_COLUMN_PARALLEL_QLINEAR(ColumnParallelQLinearGPTQMarlinImpl);
  }
  if (boost::iequals(quant_args.quant_method(), "awq") ||
      boost::iequals(quant_args.quant_method(), "GEMM")) {
    // default to use awq implementation for gemm
    // return MAKE_COLUMN_PARALLEL_QLINEAR(ColumnParallelQLinearAWQImpl);
    return MAKE_COLUMN_PARALLEL_QLINEAR(ColumnParallelQLinearAWQMarlinImpl);
  }
  // not supported quant method
  LOG(FATAL) << "Unsupported quant method: " << quant_args.quant_method();
}

std::shared_ptr<ParallelLinearImpl> create_row_parallel_qlinear(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    bool input_is_parallelized,
    const QuantArgs& quant_args,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options) {
  if (auto qlinear = create_row_parallel_qlinear_by_impl(in_features,
                                                         out_features,
                                                         bias,
                                                         input_is_parallelized,
                                                         quant_args,
                                                         parallel_args,
                                                         options)) {
    return qlinear;
  }
  if (boost::iequals(quant_args.quant_method(), "gptq")) {
    // default to use marlin implementation for gptq
    return MAKE_ROW_PARALLEL_QLINEAR(RowParallelQLinearGPTQMarlinImpl);
  }
  if (boost::iequals(quant_args.quant_method(), "awq") ||
      boost::iequals(quant_args.quant_method(), "GEMM")) {
    // default to use awq implementation for gemm
    // return MAKE_ROW_PARALLEL_QLINEAR(RowParallelQLinearAWQImpl);
    return MAKE_ROW_PARALLEL_QLINEAR(RowParallelQLinearAWQMarlinImpl);
  }
  // not supported quant method
  LOG(FATAL) << "Unsupported quant method: " << quant_args.quant_method();
}

std::shared_ptr<ParallelLinearImpl> create_column_parallel_linear(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    bool gather_output,
    const QuantArgs& quant_args,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options) {
  if (!quant_args.quant_method().empty()) {
    return create_column_parallel_qlinear(in_features,
                                          out_features,
                                          bias,
                                          gather_output,
                                          quant_args,
                                          parallel_args,
                                          options);
  }
  return MAKE_COLUMN_PARALLEL_LINEAR(ColumnParallelLinearImpl);
}

std::shared_ptr<ParallelLinearImpl> create_row_parallel_linear(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    bool input_is_parallelized,
    const QuantArgs& quant_args,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options) {
  if (!quant_args.quant_method().empty()) {
    return create_row_parallel_qlinear(in_features,
                                       out_features,
                                       bias,
                                       input_is_parallelized,
                                       quant_args,
                                       parallel_args,
                                       options);
  }
  return MAKE_ROW_PARALLEL_LINEAR(RowParallelLinearImpl);
}
}  // namespace

// construct a ColumnParallelLinear.
// chose right implementation based on the args.
ColumnParallelLinear::ColumnParallelLinear(int64_t in_features,
                                           int64_t out_features,
                                           bool bias,
                                           bool gather_output,
                                           const QuantArgs& quant_args,
                                           const ParallelArgs& parallel_args,
                                           const torch::TensorOptions& options)
    : ModuleHolder(create_column_parallel_linear(in_features,
                                                 out_features,
                                                 bias,
                                                 gather_output,
                                                 quant_args,
                                                 parallel_args,
                                                 options)) {}

ColumnParallelLinear::ColumnParallelLinear(int64_t in_features,
                                           int64_t out_features,
                                           bool bias,
                                           bool gather_output,
                                           const ParallelArgs& parallel_args,
                                           const torch::TensorOptions& options)
    : ModuleHolder(create_column_parallel_linear(in_features,
                                                 out_features,
                                                 bias,
                                                 gather_output,
                                                 {}, /*quant_args*/
                                                 parallel_args,
                                                 options)) {}

// construct a rotary positional embedding.
// chose right implementation based on the args.
RowParallelLinear::RowParallelLinear(int64_t in_features,
                                     int64_t out_features,
                                     bool bias,
                                     bool input_is_parallelized,
                                     const QuantArgs& quant_args,
                                     const ParallelArgs& parallel_args,
                                     const torch::TensorOptions& options)
    : ModuleHolder(create_row_parallel_linear(in_features,
                                              out_features,
                                              bias,
                                              input_is_parallelized,
                                              quant_args,
                                              parallel_args,
                                              options)) {}
}  // namespace llm
