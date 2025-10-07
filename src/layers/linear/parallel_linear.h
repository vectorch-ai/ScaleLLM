#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include <memory>

#include "linear.h"
#include "module/module_holder.h"

namespace llm {

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

// Fused linear layer with column parallelism.
class FusedColumnParallelLinearImpl : public MultiParallelLinearImpl {
 public:
  FusedColumnParallelLinearImpl(int64_t in_features,
                                const std::vector<int64_t>& out_features,
                                const std::vector<std::string>& prefixes,
                                bool bias,
                                bool gather_output,
                                const ParallelArgs& parallel_args,
                                const torch::TensorOptions& options);

  std::vector<torch::Tensor> forward(torch::Tensor input) override;

  // return the weight (for testing)
  torch::Tensor weight() const { return weight_; }

 private:
  // parameter members, must be registered
  // we allocate the transpose since linear performs XA^T.
  // A^T: [out_features_per_partition, in_features]
  torch::Tensor weight_;
  torch::Tensor bias_;

  std::vector<int64_t> split_sizes_;

  // whether to gather the output
  bool gather_output_;

  // parallel args
  ParallelArgs parallel_args_;
};
LLM_MODULE(FusedColumnParallelLinear);

class GroupedColumnParallelLinearImpl : public MultiParallelLinearImpl {
 public:
  GroupedColumnParallelLinearImpl(int64_t in_features,
                                  const std::vector<int64_t>& out_features,
                                  const std::vector<std::string>& prefixes,
                                  bool bias,
                                  bool gather_output,
                                  const ParallelArgs& parallel_args,
                                  const torch::TensorOptions& options);

  std::vector<torch::Tensor> forward(torch::Tensor input) override;

 private:
  // parameter members, must be registered
  std::vector<std::shared_ptr<ColumnParallelLinearImpl>> parallel_linears_;
};
LLM_MODULE(GroupedColumnParallelLinear);

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
}  // namespace llm
