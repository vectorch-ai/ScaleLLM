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
                               const torch::ScalarType& dtype,
                               const torch::Device& device);

  torch::Tensor forward(torch::Tensor input) const override;

  void pretty_print(std::ostream& stream) const override {
    stream << name() << " qweight=" << qweight_.sizes()
           << " qzeros=" << qzeros_.sizes() << " scales=" << scales_.sizes()
           << " device=" << qweight_.device();
  }

 private:
  // quantization parameters
  int64_t bits_ = 0;
  int64_t group_size_ = 0;
  int pack_factor_ = 0;

  // parallel args
  ParallelArgs parallel_args_;

  // whether to gather the output
  bool gather_output_;
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
                            const torch::ScalarType& dtype,
                            const torch::Device& device);

  torch::Tensor forward(torch::Tensor input) const override;

  void pretty_print(std::ostream& stream) const override {
    stream << name() << " qweight=" << qweight_.sizes()
           << " qzeros=" << qzeros_.sizes() << " scales=" << scales_.sizes()
           << " device=" << qweight_.device();
  }

 private:
  // quantization parameters
  int64_t bits_ = 0;
  int64_t group_size_ = 0;
  int pack_factor_ = 0;

  // parallel args
  ParallelArgs parallel_args_;

  // whether the input is already parallelized
  bool input_is_parallelized_;
};
}  // namespace llm
