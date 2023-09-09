#pragma once

#include <torch/torch.h>

#include <optional>

#include "layers/linear.h"
#include "models/model_args.h"
#include "models/parallel_args.h"

namespace llm {

class FeedForwardImpl : public torch::nn::Module {
 public:
  FeedForwardImpl(const ModelArgs& args,
                  const ParallelArgs& parallel_args,
                  const torch::ScalarType& dtype,
                  const torch::Device& device) {
    const int64_t dim = args.dim();
    const int64_t multiple_of = args.multiple_of();
    const float ffn_dim_multiplier = args.ffn_dim_multiplier().value_or(1.0f);
    int64_t hidden_dim = 4 * dim;
    hidden_dim = 2 * hidden_dim / 3;
    // custom dim factor multiplier
    hidden_dim *= ffn_dim_multiplier;
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) / multiple_of);

    // register the weight parameter
    w1_ = register_module(
        "w1",
        ColumnParallelLinear(dim, hidden_dim, parallel_args, dtype, device));
    w2_ = register_module(
        "w2", RowParallelLinear(hidden_dim, dim, parallel_args, dtype, device));
    w3_ = register_module(
        "w3",
        ColumnParallelLinear(dim, hidden_dim, parallel_args, dtype, device));
  }

  torch::Tensor forward(torch::Tensor x) {
    namespace F = torch::nn::functional;
    return w2_(F::silu(w1_(x)) * w3_(x));
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    w1_->load_state_dict(state_dict.select("w1."));
    w2_->load_state_dict(state_dict.select("w2."));
    w3_->load_state_dict(state_dict.select("w3."));
  }

 private:
  // parameter members, must be registered
  ColumnParallelLinear w1_{nullptr};
  RowParallelLinear w2_{nullptr};
  ColumnParallelLinear w3_{nullptr};
};
TORCH_MODULE(FeedForward);

}  // namespace llm
