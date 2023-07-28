#include "feedforward.h"

#include <torch/nn/functional.h>

namespace llm {

FeedForwardImpl::FeedForwardImpl(int64_t dim,
                                 int64_t hidden_dim,
                                 int64_t multiple_of,
                                 std::optional<float> ffn_dim_multiplier,
                                 int64_t world_size) {
  hidden_dim = 2 * hidden_dim / 3;
  // custom dim factor multiplier
  if (ffn_dim_multiplier.has_value()) {
    hidden_dim = static_cast<int64_t>(ffn_dim_multiplier.value() * hidden_dim);
  }
  hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) / multiple_of);
  
  // register the weight parameter
  w1_ =
      register_module("w1", ColumnParallelLinear(dim, hidden_dim, world_size));
  w2_ = register_module("w2", RowParallelLinear(hidden_dim, dim, world_size));
  w3_ =
      register_module("w3", ColumnParallelLinear(dim, hidden_dim, world_size));
}

torch::Tensor FeedForwardImpl::forward(torch::Tensor x) {
  namespace F = torch::nn::functional;
  return w2_->forward(F::silu(w1_->forward(x)) * w3_->forward(x));
}

// load the weight from the checkpoint
void FeedForwardImpl::load_state_dict(const StateDict& state_dict) {
  // call each submodule's load_state_dict function
  w1_->load_state_dict(state_dict.select("w1."));
  w2_->load_state_dict(state_dict.select("w2."));
  w3_->load_state_dict(state_dict.select("w3."));
}

}  // namespace llm
