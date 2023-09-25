#pragma once

#include <torch/torch.h>

#include "model_loader/state_dict.h"
#include "models/args.h"

namespace llm {
// an interface for parallel linear layer.
// all linear classes should inherit from this class and implement the forward
// function.
class ParallelLinearImpl : public torch::nn::Module {
 public:
  ~ParallelLinearImpl() override = default;

  virtual torch::Tensor forward(torch::Tensor input) const = 0;

  virtual void load_state_dict(const StateDict& state_dict) = 0;

  virtual void verify_loaded_weights() const = 0;

  virtual void load_state_dict(
      const StateDict& /*state_dict*/,
      const std::vector<std::string_view>& /*prefixes*/) = 0;
};

class ColumnParallelLinear
    : public torch::nn::ModuleHolder<ParallelLinearImpl> {
 public:
  using torch::nn::ModuleHolder<ParallelLinearImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = ParallelLinearImpl;

  // construct a rotary positional embedding.
  // chose right implementation based on the args.
  ColumnParallelLinear(int64_t in_features,
                       int64_t out_features,
                       bool gather_output,
                       const QuantizationArgs& quant_args,
                       const ParallelArgs& parallel_args,
                       const torch::ScalarType& dtype,
                       const torch::Device& device);

  ColumnParallelLinear(int64_t in_features,
                       int64_t out_features,
                       bool gather_output,
                       const ParallelArgs& parallel_args,
                       const torch::ScalarType& dtype,
                       const torch::Device& device);
};

class RowParallelLinear : public torch::nn::ModuleHolder<ParallelLinearImpl> {
 public:
  using torch::nn::ModuleHolder<ParallelLinearImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = ParallelLinearImpl;

  // construct a rotary positional embedding.
  // chose right implementation based on the args.
  RowParallelLinear(int64_t in_features,
                    int64_t out_features,
                    bool input_is_parallelized,
                    const QuantizationArgs& quant_args,
                    const ParallelArgs& parallel_args,
                    const torch::ScalarType& dtype,
                    const torch::Device& device);
};
}  // namespace llm
