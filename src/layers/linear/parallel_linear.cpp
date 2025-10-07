#include "parallel_linear.h"

#include <c10/core/TensorImpl.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <boost/algorithm/string.hpp>
#include <memory>

#include "model_parallel/model_parallel.h"
#include "module/module.h"
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
  return nullptr;
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
  return nullptr;
}

std::shared_ptr<ParallelLinearImpl> create_column_parallel_linear(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    bool gather_output,
    const QuantArgs& quant_args,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options,
    const std::string& prefix) {
  if (!quant_args.quant_method().empty()) {
    return create_column_parallel_qlinear(in_features,
                                          out_features,
                                          bias,
                                          gather_output,
                                          quant_args,
                                          parallel_args,
                                          options);
  }
  return std ::make_shared<ColumnParallelLinearImpl>(in_features,
                                                     out_features,
                                                     bias,
                                                     gather_output,
                                                     parallel_args,
                                                     options,
                                                     prefix);
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
  return std ::make_shared<RowParallelLinearImpl>(in_features,
                                                  out_features,
                                                  bias,
                                                  input_is_parallelized,
                                                  parallel_args,
                                                  options);
}

// std::shared_ptr<MultiParallelLinearImpl> create_multi_column_parallel_linear(
//     int64_t in_features,
//     const std::vector<int64_t>& out_features,
//     const std::vector<std::string>& prefixes,
//     bool bias,
//     bool gather_output,
//     const QuantArgs& quant_args,
//     const ParallelArgs& parallel_args,
//     const torch::TensorOptions& options) {
//   // check if the linear layers can be fused
//   const bool fused = quant_args.can_be_fused();
//   std::shared_ptr<MultiParallelLinearImpl> impl;
//   if (fused) {
//     return std::make_shared<FusedColumnParallelLinearImpl>(in_features,
//                                                        out_features,
//                                                        prefixes,
//                                                        bias,
//                                                        gather_output,
//                                                        parallel_args,
//                                                        options);
//   }

//   return std::make_shared<GroupedColumnParallelLinearImpl>(in_features,
//                                                            out_features,
//                                                            prefixes,
//                                                            bias,
//                                                            gather_output,
//                                                            parallel_args,
//                                                            options);
// }
}  // namespace

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

// construct a ColumnParallelLinear.
// chose right implementation based on the args.
ColumnParallelLinear::ColumnParallelLinear(int64_t in_features,
                                           int64_t out_features,
                                           bool bias,
                                           bool gather_output,
                                           const QuantArgs& quant_args,
                                           const ParallelArgs& parallel_args,
                                           const torch::TensorOptions& options,
                                           const std::string& prefix)
    : ModuleHolder(create_column_parallel_linear(in_features,
                                                 out_features,
                                                 bias,
                                                 gather_output,
                                                 quant_args,
                                                 parallel_args,
                                                 options,
                                                 prefix)) {}

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
                                                 options,
                                                 "")) {}

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
