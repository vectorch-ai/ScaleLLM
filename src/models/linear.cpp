#include "linear.h"

#include <torch/nn/functional/embedding.h>
#include <torch/nn/module.h>
#include <torch/torch.h>

namespace llm {

ColumnParallelLinearImpl::ColumnParallelLinearImpl(int64_t in_features,
                                                   int64_t out_features,
                                                   int64_t world_size)
    : world_size_(world_size) {
  weight_ =
      register_parameter("weight", torch::empty({in_features, out_features}));
}

torch::Tensor ColumnParallelLinearImpl::forward(torch::Tensor input) {
  namespace F = torch::nn::functional;
  auto output = F::linear(input, weight_);
  if (world_size_ > 1) {
    // call all reduce or all gather with concat
    // torch::distributed::all_reduce(input_);
  }
  return output;
}

// load the weight from the checkpoint
void ColumnParallelLinearImpl::load_state_dict(const StateDict& state_dict) {
  weight_ = state_dict.get_tensor("weight");
}

RowParallelLinearImpl::RowParallelLinearImpl(int64_t in_features,
                                             int64_t out_features,
                                             int64_t world_size)
    : world_size_(world_size) {
  weight_ =
      register_parameter("weight", torch::empty({in_features, out_features}));
}

torch::Tensor RowParallelLinearImpl::forward(torch::Tensor input) {
  namespace F = torch::nn::functional;
  auto output = F::linear(input, weight_);
  if (world_size_ > 1) {
    // call all reduce or all gather with concat
    // torch::distributed::all_reduce(input_);
  }
  return output;
}

// load the weight from the checkpoint
void RowParallelLinearImpl::load_state_dict(const StateDict& state_dict) {
  weight_.copy_(state_dict.get_tensor("weight"));
}

}  // namespace llm
