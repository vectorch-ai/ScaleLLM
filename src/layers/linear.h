#pragma once

#include <ATen/core/TensorBody.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include "model_loader/state_dict.h"
#include "model_parallel.h"
#include "models/parallel_args.h"

namespace llm {

// Linear layer with column parallelism.
// The linear layer is defined as Y = XA + b. A is parallelized along
// its second dimension as A = [A_1, ..., A_p].
class ColumnParallelLinearImpl : public torch::nn::Module {
 public:
  ColumnParallelLinearImpl(int64_t in_features,
                           int64_t out_features,
                           bool gather_output,
                           const ParallelArgs& parallel_args,
                           const torch::ScalarType& dtype,
                           const torch::Device& device)
      : gather_output_(gather_output), parallel_args_(parallel_args) {
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
    if (parallel_args_.world_size() > 1 && gather_output_) {
      output = gather_from_model_parallel_region(output, parallel_args_);
    }
    return output;
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    const auto weight = state_dict.get_sharded_tensor(
        "weight",
        /*dim=*/0,
        /*rank=*/parallel_args_.rank(),
        /*world_size=*/parallel_args_.world_size());
    if (weight.defined()) {
      CHECK_EQ(weight_.sizes(), weight.sizes()) << "weight size mismatch";
      weight_.copy_(weight);
      is_loaded_list_[0] = true;
    }
  }

  // special load_state_dict for fused cases
  void load_state_dict(const StateDict& state_dict,
                       const std::vector<std::string_view>& prefixes,
                       const std::vector<int64_t>& sizes) {
    CHECK_EQ(prefixes.size(), sizes.size());
    CHECK_EQ(std::accumulate(sizes.begin(), sizes.end(), 0), weight_.size(0));

    if (is_loaded_list_.size() < prefixes.size()) {
      is_loaded_list_.resize(prefixes.size(), false);
    }

    for (size_t i = 0; i < prefixes.size(); ++i) {
      std::string name = std::string(prefixes[i]) + "weight";
      const auto weight = state_dict.get_sharded_tensor(
          name,
          /*dim=*/0,
          /*rank=*/parallel_args_.rank(),
          /*world_size=*/parallel_args_.world_size());
      if (weight.defined()) {
        int64_t end = 0;
        for (size_t j = 0; j <= i; ++j) {
          end += sizes[j];
        }
        const int64_t start = end - sizes[i];
        weight_.index_put_(
            {torch::indexing::Slice(start, end), torch::indexing::Slice()},
            weight);
        is_loaded_list_[i] = true;
      }
    }
  }

  // whether the weight is loaded
  bool is_loaded() const {
    // all the weights are loaded
    return std::all_of(is_loaded_list_.begin(),
                       is_loaded_list_.end(),
                       [](bool v) { return v; });
  }

  void pretty_print(std::ostream& stream) const override {
    stream << name() << " " << weight_.sizes() << " " << weight_.device();
  }

  // return the weight (for testing)
  torch::Tensor weight() const { return weight_; }

 private:
  // parameter members, must be registered
  // we allocate the transpose since linear performs XA^T.
  // A^T: [out_features_per_partition, in_features]
  torch::Tensor weight_{nullptr};

  std::vector<bool> is_loaded_list_{false};

  // parallel args
  ParallelArgs parallel_args_;

  // whether to gather the output
  bool gather_output_;
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
                        bool input_is_parallelized,
                        const ParallelArgs& parallel_args,
                        const torch::ScalarType& dtype,
                        const torch::Device& device)
      : input_is_parallelized_(input_is_parallelized),
        parallel_args_(parallel_args) {
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
    if (!input_is_parallelized_) {
      input = scatter_to_model_parallel_region(input, parallel_args_);
    }
    auto output = F::linear(input, weight_);
    if (parallel_args_.world_size() > 1) {
      output = reduce_from_model_parallel_region(output, parallel_args_);
    }
    return output;
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    const auto weight = state_dict.get_sharded_tensor(
        "weight",
        /*dim=*/1,
        /*rank=*/parallel_args_.rank(),
        /*world_size=*/parallel_args_.world_size());
    if (weight.defined()) {
      CHECK_EQ(weight_.sizes(), weight.sizes()) << "weight size mismatch";
      weight_.copy_(weight);
      is_loaded_ = true;
    }
  }

  // whether the weight is loaded
  bool is_loaded() const { return is_loaded_; }

  void pretty_print(std::ostream& stream) const override {
    stream << name() << " " << weight_.sizes() << " " << weight_.device();
  }

  // return the weight (for testing)
  torch::Tensor weight() const { return weight_; }

 private:
  // parameter members, must be registered
  // we allocate the transpose since linear performs XA^T.
  // A^T: [out_features, in_features_per_partition]
  torch::Tensor weight_{nullptr};

  // whether the weight is loaded
  bool is_loaded_ = false;

  // parallel args
  ParallelArgs parallel_args_;

  // whether the input is already parallelized
  bool input_is_parallelized_;
};
TORCH_MODULE(RowParallelLinear);
}  // namespace llm
