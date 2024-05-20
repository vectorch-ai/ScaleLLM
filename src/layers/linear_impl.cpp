#include "linear_impl.h"

#include <c10/core/TensorImpl.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>

#include "model_loader/state_dict.h"
#include "model_parallel/model_parallel.h"

namespace llm {
namespace detail {

void merge_weights(const std::string& tensor_name,
                   std::vector<torch::Tensor> weight_list,
                   int64_t dim,
                   bool clone,
                   std::vector<torch::Tensor>& accumulated_weight_list,
                   torch::Tensor& weight,
                   bool& weight_is_loaded) {
  // return if the weight is already loaded
  if (weight_is_loaded) {
    return;
  }
  // resize the accumulated weight list if needed
  if (accumulated_weight_list.size() < weight_list.size()) {
    accumulated_weight_list.resize(weight_list.size());
  }

  // copy over accumulated weights
  for (size_t i = 0; i < weight_list.size(); ++i) {
    if (accumulated_weight_list[i].defined()) {
      CHECK(!weight_list[i].defined()) << tensor_name << " weight already set";
      weight_list[i] = accumulated_weight_list[i];
    }
  }

  const bool all_loaded = std::all_of(
      weight_list.begin(), weight_list.end(), [](const torch::Tensor& t) {
        return t.defined();
      });
  if (!all_loaded) {
    // accumulate the weights for future merge
    for (size_t i = 0; i < weight_list.size(); ++i) {
      if (!accumulated_weight_list[i].defined() && weight_list[i].defined()) {
        accumulated_weight_list[i] =
            clone ? weight_list[i].clone() : weight_list[i];
      }
    }
  } else {
    const auto merged_weight = torch::cat(weight_list, /*dim=*/dim);
    CHECK_EQ(weight.sizes(), merged_weight.sizes())
        << "weight size mismatch for " << tensor_name;
    weight.copy_(merged_weight);
    // release the memory for weight_list
    accumulated_weight_list.clear();
    weight_is_loaded = true;
  }
}

}  // namespace detail

// Linear layer with column parallelism.
ColumnParallelLinearImpl::ColumnParallelLinearImpl(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    bool gather_output,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options)
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
      torch::empty({out_features_per_partition, in_features}, options),
      /*requires_grad=*/false);

  if (bias) {
    bias_ =
        register_parameter("bias",
                           torch::empty({out_features_per_partition}, options),
                           /*requires_grad=*/false);
  }
}

torch::Tensor ColumnParallelLinearImpl::forward(torch::Tensor input) const {
  namespace F = torch::nn::functional;
  auto output = F::linear(input, weight_, bias_);
  if (parallel_args_.world_size() > 1 && gather_output_) {
    output = gather_from_model_parallel_region(output, parallel_args_);
  }
  return output;
}

// load the weight from the checkpoint
void ColumnParallelLinearImpl::load_state_dict(const StateDict& state_dict) {
  // call load_state_dict with identity transform
  load_state_dict(state_dict,
                  [](const torch::Tensor& tensor) { return tensor; });
}

void ColumnParallelLinearImpl::load_state_dict(const StateDict& state_dict,
                                               TensorTransform transform_func) {
  CHECK(transform_func != nullptr) << "transform_func must be provided";
  auto weight =
      state_dict.get_sharded_tensor("weight",
                                    /*dim=*/0,
                                    /*rank=*/parallel_args_.rank(),
                                    /*world_size=*/parallel_args_.world_size());
  if (weight.defined()) {
    weight = transform_func(weight);
    CHECK_EQ(weight_.sizes(), weight.sizes())
        << "weight size mismatch for " << name();
    weight_.copy_(weight);
    weight_is_loaded_ = true;
  }

  if (bias_.defined()) {
    auto bias = state_dict.get_sharded_tensor(
        "bias",
        /*dim=*/0,
        /*rank=*/parallel_args_.rank(),
        /*world_size=*/parallel_args_.world_size());
    if (bias.defined()) {
      bias = transform_func(bias);
      CHECK_EQ(bias_.sizes(), bias.sizes())
          << "bias size mismatch for " << name();
      bias_.copy_(bias);
      bias_is_loaded_ = true;
    }
  }
}

// special load_state_dict for fused cases
void ColumnParallelLinearImpl::load_state_dict(
    const StateDict& state_dict,
    const std::vector<std::string>& prefixes) {
  std::vector<torch::Tensor> weight_list(prefixes.size());
  std::vector<torch::Tensor> bias_list(prefixes.size());
  for (size_t i = 0; i < prefixes.size(); ++i) {
    std::string name = std::string(prefixes[i]) + "weight";
    const auto weight =
        state_dict.get_sharded_tensor(name,
                                      /*dim=*/0,
                                      parallel_args_.rank(),
                                      parallel_args_.world_size());
    if (weight.defined()) {
      CHECK(!weight_list[i].defined()) << "weight already loaded";
      weight_list[i] = weight;
    }

    if (bias_.defined()) {
      name = std::string(prefixes[i]) + "bias";
      const auto bias =
          state_dict.get_sharded_tensor(name,
                                        /*dim=*/0,
                                        parallel_args_.rank(),
                                        parallel_args_.world_size());
      if (bias.defined()) {
        CHECK(!bias_list[i].defined()) << "bias already loaded";
        bias_list[i] = bias;
      }
    }
  }
  detail::merge_weights(name(),
                        std::move(weight_list),
                        /*dim=*/0,
                        /*clone=*/true,
                        weight_list_,
                        weight_,
                        weight_is_loaded_);
  if (bias_.defined()) {
    detail::merge_weights(name(),
                          std::move(bias_list),
                          /*dim=*/0,
                          /*clone=*/true,
                          bias_list_,
                          bias_,
                          bias_is_loaded_);
  }
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
  const auto world_size = parallel_args_.world_size();
  CHECK(in_features % world_size == 0)
      << "in_features " << in_features << " not divisible by world_size "
      << world_size;
  const int64_t in_features_per_partition = in_features / world_size;
  // Allocate the transpose since linear performs XA^T.
  weight_ = register_parameter(
      "weight",
      torch::empty({out_features, in_features_per_partition}, options),
      /*requires_grad=*/false);

  if (bias) {
    bias_ = register_parameter("bias",
                               torch::empty({out_features}, options),
                               /*requires_grad=*/false);
  }
}

torch::Tensor RowParallelLinearImpl::forward(torch::Tensor input) const {
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

// load the weight from the checkpoint
void RowParallelLinearImpl::load_state_dict(const StateDict& state_dict) {
  const auto weight =
      state_dict.get_sharded_tensor("weight",
                                    /*dim=*/1,
                                    /*rank=*/parallel_args_.rank(),
                                    /*world_size=*/parallel_args_.world_size());
  if (weight.defined()) {
    CHECK_EQ(weight_.sizes(), weight.sizes())
        << "weight size mismatch for " << name();
    weight_.copy_(weight);
    weight_is_loaded_ = true;
  }

  if (bias_.defined()) {
    const auto bias = state_dict.get_tensor("bias");
    if (bias.defined()) {
      CHECK_EQ(bias_.sizes(), bias.sizes())
          << "bias size mismatch for " << name();
      bias_.copy_(bias);
      bias_is_loaded_ = true;
    }
  }
}

}  // namespace llm
