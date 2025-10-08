#include "qlinear_awq_impl.h"

#include <glog/logging.h>
#include <torch/torch.h>
#include <torch/types.h>

extern torch::Tensor gemm_forward_cuda(torch::Tensor in_feats,
                                       torch::Tensor kernel,
                                       torch::Tensor scaling_factors,
                                       torch::Tensor zeros,
                                       int split_k_iters);

namespace llm {
ColumnParallelQLinearAWQImpl::ColumnParallelQLinearAWQImpl(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    const QuantArgs& quant_args,
    bool gather_output,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options)
    : ColumnParallelQLinearImpl(in_features,
                                out_features,
                                bias,
                                quant_args,
                                /*qweight_pack_dim=*/1,
                                gather_output,
                                parallel_args,
                                options) {
  const auto bits = quant_args.bits();
  const auto group_size = quant_args.group_size();
  CHECK(bits == 4) << "Only 4 bits are supported for AWQ";
  pack_factor_ = 32 / bits;
}

torch::Tensor ColumnParallelQLinearAWQImpl::quant_matmul(
    const torch::Tensor& input,
    const torch::Tensor& qweight,
    const torch::Tensor& qzeros,
    const torch::Tensor& scales) const {
  const int64_t out_features = qweight.size(-1) * pack_factor_;
  torch::Tensor output =
      gemm_forward_cuda(input, qweight, scales, qzeros, pack_factor_);
  output = output.view({-1, out_features});
  return output;
}

RowParallelQLinearAWQImpl::RowParallelQLinearAWQImpl(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    const QuantArgs& quant_args,
    bool input_is_parallelized,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options)
    : RowParallelQLinearImpl(in_features,
                             out_features,
                             bias,
                             quant_args,
                             /*qweight_pack_dim=*/1,
                             input_is_parallelized,
                             parallel_args,
                             options) {
  const auto bits = quant_args.bits();
  CHECK(bits == 4) << "Only 4 bits are supported for AWQ";
  pack_factor_ = 32 / bits;
}

torch::Tensor RowParallelQLinearAWQImpl::quant_matmul(
    const torch::Tensor& input,
    const torch::Tensor& qweight,
    const torch::Tensor& qzeros,
    const torch::Tensor& scales) const {
  const int64_t out_features = qweight.size(-1) * pack_factor_;
  torch::Tensor output =
      gemm_forward_cuda(input, qweight, scales, qzeros, pack_factor_);
  output = output.view({-1, out_features});
  return output;
}

}  // namespace llm
