#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include "model_loader/state_dict.h"
#include "model_parallel/parallel_args.h"
#include "module/module.h"
#include "module/module_holder.h"
#include "quantization/quant_args.h"

namespace llm {

using TensorTransform = std::function<torch::Tensor(const torch::Tensor&)>;

// an interface for parallel linear layer.
// all linear classes should inherit from this class and implement the forward
// function.
class ParallelLinearImpl : public Module {
 public:
  ~ParallelLinearImpl() override = default;

  virtual torch::Tensor forward(torch::Tensor input) = 0;

  virtual void load_state_dict(const StateDict& state_dict) = 0;

  virtual void verify_loaded_weights(const std::string& prefix = "") const = 0;

  // load state dict with a transform function
  virtual void load_state_dict(const StateDict& /*state_dict*/,
                               TensorTransform /*transform_func*/) {
    LOG(FATAL) << "not implemented";
  }

  // special load_state_dict for fused cases
  virtual void load_state_dict(const StateDict& /*state_dict*/,
                               const std::vector<std::string>& /*prefixes*/) {
    LOG(FATAL) << "not implemented";
  }
};

class ColumnParallelLinear : public ModuleHolder<ParallelLinearImpl> {
 public:
  using ModuleHolder<ParallelLinearImpl>::ModuleHolder;
  using Impl [[maybe_unused]] = ParallelLinearImpl;

  // construct a rotary positional embedding.
  // chose right implementation based on the args.
  ColumnParallelLinear(int64_t in_features,
                       int64_t out_features,
                       bool bias,
                       bool gather_output,
                       const QuantArgs& quant_args,
                       const ParallelArgs& parallel_args,
                       const torch::TensorOptions& options);

  ColumnParallelLinear(int64_t in_features,
                       int64_t out_features,
                       bool bias,
                       bool gather_output,
                       const ParallelArgs& parallel_args,
                       const torch::TensorOptions& options);
};

class RowParallelLinear : public ModuleHolder<ParallelLinearImpl> {
 public:
  using ModuleHolder<ParallelLinearImpl>::ModuleHolder;
  using Impl [[maybe_unused]] = ParallelLinearImpl;

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
