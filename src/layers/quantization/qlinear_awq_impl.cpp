#include "qlinear_awq_impl.h"

#include <gflags/gflags.h>
#include <torch/torch.h>
#include <torch/types.h>

#include "../model_parallel.h"
#include "common/logging.h"
#include "model_loader/state_dict.h"
#include "models/args.h"

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
    const QuantizationArgs& quant_args,
    bool gather_output,
    const ParallelArgs& parallel_args,
    torch::ScalarType dtype,
    const torch::Device& device)
    : ColumnParallelQLinearImpl(in_features,
                                out_features,
                                bias,
                                quant_args,
                                /*qweight_pack_dim=*/1,
                                gather_output,
                                parallel_args,
                                dtype,
                                device) {
  const auto bits = quant_args.bits();
  const auto group_size = quant_args.group_size();
  GCHECK(bits == 4) << "Only 4 bits are supported for AWQ";
  GCHECK(group_size > 0) << "group_size must be positive";
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
    const QuantizationArgs& quant_args,
    bool input_is_parallelized,
    const ParallelArgs& parallel_args,
    torch::ScalarType dtype,
    const torch::Device& device)
    : RowParallelQLinearImpl(in_features,
                             out_features,
                             bias,
                             quant_args,
                             /*qweight_pack_dim=*/1,
                             input_is_parallelized,
                             parallel_args,
                             dtype,
                             device) {
  const auto bits = quant_args.bits();
  const auto group_size = quant_args.group_size();
  GCHECK(bits == 4) << "Only 4 bits are supported for AWQ";
  GCHECK(group_size > 0) << "group_size must be positive";
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
