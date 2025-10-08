#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include "layers/module/module.h"
#include "layers/module/module_holder.h"
#include "layers/quantization/quant_args.h"
#include "model_loader/state_dict.h"
#include "model_parallel/parallel_args.h"

namespace llm {

// an interface for parallel linear layer.
// all linear classes should inherit from this class and implement the forward
// function.
class ParallelLinearImpl : public Module {
 public:
  ~ParallelLinearImpl() override = default;

  virtual torch::Tensor forward(torch::Tensor input) = 0;
};

// Linear layer with column parallelism.
// The linear layer is defined as Y = XA + b. A is parallelized along
// its second dimension as A = [A_1, ..., A_p].
class ColumnParallelLinearImpl : public ParallelLinearImpl {
 public:
  ColumnParallelLinearImpl(int64_t in_features,
                           int64_t out_features,
                           bool bias,
                           bool gather_output,
                           const ParallelArgs& parallel_args,
                           const torch::TensorOptions& options,
                           const std::string& prefix = "");

  torch::Tensor forward(torch::Tensor input) override;

  // return the weight (for testing)
  torch::Tensor weight() const { return weight_; }

 private:
  // parameter members, must be registered
  // we allocate the transpose since linear performs XA^T.
  // A^T: [out_features_per_partition, in_features]
  torch::Tensor weight_;
  torch::Tensor bias_;

  // whether to gather the output
  bool gather_output_;

  // parallel args
  ParallelArgs parallel_args_;
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
class RowParallelLinearImpl : public ParallelLinearImpl {
 public:
  RowParallelLinearImpl(int64_t in_features,
                        int64_t out_features,
                        bool bias,
                        bool input_is_parallelized,
                        const ParallelArgs& parallel_args,
                        const torch::TensorOptions& options);

  torch::Tensor forward(torch::Tensor input) override;

  // return the weight (for testing)
  torch::Tensor weight() const { return weight_; }

 private:
  // parameter members, must be registered
  // we allocate the transpose since linear performs XA^T.
  // A^T: [out_features, in_features_per_partition]
  torch::Tensor weight_;
  torch::Tensor bias_;

  // whether the input is already parallelized
  bool input_is_parallelized_;

  // parallel args
  ParallelArgs parallel_args_;
};

class ColumnParallelLinear : public ModuleHolder<ParallelLinearImpl> {
 public:
  /* implicit */ ColumnParallelLinear(std::nullptr_t);

  /* implicit */ ColumnParallelLinear(
      std::shared_ptr<ParallelLinearImpl> module);

  // construct a rotary positional embedding.
  // chose right implementation based on the args.
  ColumnParallelLinear(int64_t in_features,
                       int64_t out_features,
                       bool bias,
                       bool gather_output,
                       const QuantArgs& quant_args,
                       const ParallelArgs& parallel_args,
                       const torch::TensorOptions& options,
                       const std::string& prefix = "");

  ColumnParallelLinear(int64_t in_features,
                       int64_t out_features,
                       bool bias,
                       bool gather_output,
                       const ParallelArgs& parallel_args,
                       const torch::TensorOptions& options);
};

class RowParallelLinear : public ModuleHolder<ParallelLinearImpl> {
 public:
  /* implicit */ RowParallelLinear(std::nullptr_t);

  /* implicit */ RowParallelLinear(std::shared_ptr<ParallelLinearImpl> module);

  // construct a rotary positional embedding.
  // chose right implementation based on the args.
  RowParallelLinear(int64_t in_features,
                    int64_t out_features,
                    bool bias,
                    bool input_is_parallelized,
                    const QuantArgs& quant_args,
                    const ParallelArgs& parallel_args,
                    const torch::TensorOptions& options);
};

}  // namespace llm
