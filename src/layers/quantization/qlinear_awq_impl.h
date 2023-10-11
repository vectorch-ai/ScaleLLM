#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include "model_loader/state_dict.h"
#include "models/args.h"
#include "qlinear_impl.h"

namespace llm {
// quantized linear layers using awq

// Quantized Linear layer with column parallelism.
// The linear layer is defined as Y = XA + b. A is parallelized along
// its second dimension as A = [A_1, ..., A_p].
class ColumnParallelQLinearAWQImpl : public ColumnParallelQLinearImpl {
 public:
  ColumnParallelQLinearAWQImpl(int64_t in_features,
                               int64_t out_features,
                               bool bias,
                               int64_t bits,
                               int64_t group_size,
                               bool gather_output,
                               const ParallelArgs& parallel_args,
                               torch::ScalarType dtype,
                               const torch::Device& device);

  torch::Tensor quant_matmul(const torch::Tensor& input,
                             const torch::Tensor& qweight,
                             const torch::Tensor& qzeros,
                             const torch::Tensor& scales) const override;

 private:
  // quantization parameters
  int64_t bits_ = 0;
  int64_t group_size_ = 0;
  int pack_factor_ = 0;
};

// Linear layer with row parallelism.
//     The linear layer is defined as Y = XA + b. A is parallelized along
//     its first dimension and X along its second dimension as:
//                -   -
//               | A_1 |
//               | .   |
//           A = | .   |       X = [X_1, ..., X_p]
//               | .   |
//               | A_p |
//                -   -
class RowParallelQLinearAWQImpl : public RowParallelQLinearImpl {
 public:
  RowParallelQLinearAWQImpl(int64_t in_features,
                            int64_t out_features,
                            bool bias,
                            int64_t bits,
                            int64_t group_size,
                            bool input_is_parallelized,
                            const ParallelArgs& parallel_args,
                            torch::ScalarType dtype,
                            const torch::Device& device);

  torch::Tensor quant_matmul(const torch::Tensor& input,
                             const torch::Tensor& qweight,
                             const torch::Tensor& qzeros,
                             const torch::Tensor& scales) const override;

 private:
  // quantization parameters
  int64_t bits_ = 0;
  int64_t group_size_ = 0;
  int pack_factor_ = 0;
};
}  // namespace llm
