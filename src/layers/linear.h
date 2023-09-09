#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include "models/parallel_args.h"
#include "torch_utils/state_dict.h"

namespace llm {

// Linear layer with column parallelism.
// The linear layer is defined as Y = XA + b. A is parallelized along
// its second dimension as A = [A_1, ..., A_p].
class ColumnParallelLinearImpl : public torch::nn::Module {
 public:
  ColumnParallelLinearImpl(int64_t in_features,
                           int64_t out_features,
                           const ParallelArgs& parallel_args,
                           const torch::ScalarType& dtype,
                           const torch::Device& device)
      : parallel_args_(parallel_args) {
    const auto world_size = parallel_args_.world_size();
    CHECK(out_features % world_size == 0)
        << "out_features " << out_features << " not divisible by world_size "
        << world_size;
    const int64_t out_features_per_partition = out_features / world_size;

    // Note: torch.nn.functional.linear performs XA^T + b and as a result
    // we allocate the transpose.
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features_per_partition, in_features},
                     torch::dtype(dtype).device(device)),
        /*requires_grad=*/false);
  }

  torch::Tensor forward(torch::Tensor input) {
    namespace F = torch::nn::functional;
    auto output = F::linear(input, weight_);
    if (parallel_args_.world_size() > 1) {
      // call all reduce or all gather with concat
      // torch::distributed::all_reduce(input_);
    }
    return output;
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    const auto weight = state_dict.get_tensor("weight");
    if (weight.defined()) {
      CHECK_EQ(weight_.sizes(), weight.sizes()) << "weight size mismatch";
      weight_.copy_(weight);
    } else {
      LOG(WARNING) << "weight is not defined";
    }
  }

  void pretty_print(std::ostream& stream) const override {
    stream << name() << " " << weight_.sizes() << " " << weight_.device();
  }

 private:
  // parameter members, must be registered
  // we allocate the transpose since linear performs XA^T.
  // A^T: [out_features_per_partition, in_features]
  torch::Tensor weight_{nullptr};

  // parallel args
  ParallelArgs parallel_args_;
};
TORCH_MODULE(ColumnParallelLinear);

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
class RowParallelLinearImpl : public torch::nn::Module {
 public:
  RowParallelLinearImpl(int64_t in_features,
                        int64_t out_features,
                        const ParallelArgs& parallel_args,
                        const torch::ScalarType& dtype,
                        const torch::Device& device)
      : parallel_args_(parallel_args) {
    const auto world_size = parallel_args_.world_size();
    CHECK(in_features % world_size == 0)
        << "in_features " << in_features << " not divisible by world_size "
        << world_size;
    const int64_t in_features_per_partition = in_features / world_size;
    // Allocate the transpose since linear performs XA^T.
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features, in_features_per_partition},
                     torch::dtype(dtype).device(device)),
        /*requires_grad=*/false);
  }

  torch::Tensor forward(torch::Tensor input) {
    namespace F = torch::nn::functional;
    auto output = F::linear(input, weight_);
    if (parallel_args_.world_size() > 1) {
      // call all reduce or all gather with concat
      // torch::distributed::all_reduce(input_);
    }
    return output;
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    const auto weight = state_dict.get_tensor("weight");
    if (weight.defined()) {
      CHECK_EQ(weight_.sizes(), weight.sizes()) << "weight size mismatch";
      weight_.copy_(weight);
    } else {
      LOG(WARNING) << "weight is not defined";
    }
  }

  void pretty_print(std::ostream& stream) const override {
    stream << name() << " " << weight_.sizes() << " " << weight_.device();
  }

 private:
  // parameter members, must be registered
  // we allocate the transpose since linear performs XA^T.
  // A^T: [out_features, in_features_per_partition]
  torch::Tensor weight_{nullptr};

  // parallel args
  ParallelArgs parallel_args_;
};
TORCH_MODULE(RowParallelLinear);
}  // namespace llm
