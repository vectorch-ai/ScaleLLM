#include "qlinear_awq_impl.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <torch/torch.h>
#include <torch/types.h>

#include "../model_parallel.h"
#include "model_loader/state_dict.h"
#include "models/args.h"

extern torch::Tensor gemm_forward_cuda(torch::Tensor _in_feats,
                                       torch::Tensor _kernel,
                                       torch::Tensor _scaling_factors,
                                       torch::Tensor _zeros,
                                       int split_k_iters);

namespace llm {
ColumnParallelQLinearAWQImpl::ColumnParallelQLinearAWQImpl(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    int64_t bits,
    int64_t group_size,
    bool gather_output,
    const ParallelArgs& parallel_args,
    const torch::ScalarType& dtype,
    const torch::Device& device)
    : ColumnParallelQLinearImpl(in_features,
                                out_features,
                                bias,
                                bits,
                                group_size,
                                /*qweight_pack_dim=*/1,
                                parallel_args.rank(),
                                parallel_args.world_size(),
                                dtype,
                                device),
      bits_(bits),
      group_size_(group_size),
      parallel_args_(parallel_args),
      gather_output_(gather_output) {
  CHECK(bits == 4) << "Only 4 bits are supported for AWQ";
  CHECK(group_size > 0) << "group_size must be positive";
  pack_factor_ = 32 / bits;
}

torch::Tensor ColumnParallelQLinearAWQImpl::forward(torch::Tensor input) const {
  const int64_t out_features = qweight_.size(-1) * pack_factor_;
  torch::Tensor output =
      gemm_forward_cuda(input, qweight_, scales_, qzeros_, pack_factor_);
  output = output.reshape({-1, out_features});
  if (bias_.defined()) {
    output.add_(bias_);
  }
  if (parallel_args_.world_size() > 1 && gather_output_) {
    output = gather_from_model_parallel_region(output, parallel_args_);
  }
  return output;
}

RowParallelQLinearAWQImpl::RowParallelQLinearAWQImpl(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    int64_t bits,
    int64_t group_size,
    bool input_is_parallelized,
    const ParallelArgs& parallel_args,
    const torch::ScalarType& dtype,
    const torch::Device& device)
    : RowParallelQLinearImpl(in_features,
                             out_features,
                             bias,
                             bits,
                             group_size,
                             /*qweight_pack_dim=*/1,
                             parallel_args.rank(),
                             parallel_args.world_size(),
                             dtype,
                             device),
      bits_(bits),
      group_size_(group_size),
      parallel_args_(parallel_args),
      input_is_parallelized_(input_is_parallelized) {
  CHECK(bits == 4) << "Only 4 bits are supported for AWQ";
  CHECK(group_size > 0) << "group_size must be positive";
  pack_factor_ = 32 / bits;
}

torch::Tensor RowParallelQLinearAWQImpl::forward(torch::Tensor input) const {
  if (!input_is_parallelized_) {
    input = scatter_to_model_parallel_region(input, parallel_args_);
  }
  const int64_t out_features = qweight_.size(-1) * pack_factor_;
  torch::Tensor output =
      gemm_forward_cuda(input, qweight_, scales_, qzeros_, pack_factor_);
  output = output.reshape({-1, out_features});
  if (bias_.defined()) {
    output.add_(bias_);
  }
  if (parallel_args_.world_size() > 1) {
    output = reduce_from_model_parallel_region(output, parallel_args_);
  }
  return output;
}

}  // namespace llm
