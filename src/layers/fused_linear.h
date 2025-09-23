#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include "linear.h"
#include "model_loader/state_dict.h"
#include "model_parallel/parallel_args.h"
#include "module/module.h"
#include "module/module_holder.h"
#include "quantization/quant_args.h"

namespace llm {

class LegacyFusedColumnParallelLinearImpl : public Module {
 public:
  LegacyFusedColumnParallelLinearImpl(int64_t in_features,
                                      const std::vector<int64_t>& out_features,
                                      bool bias,
                                      bool gather_output,
                                      const QuantArgs& quant_args,
                                      const ParallelArgs& parallel_args,
                                      const torch::TensorOptions& options);

  std::vector<torch::Tensor> forward(torch::Tensor input);

  // load_state_dict for fused weights
  void load_state_dict(const StateDict& state_dict,
                       const std::vector<std::string>& prefixes);

  void verify_loaded_weights(const std::string& prefix = "") const;

  // whether the linear layer is fused
  bool fused() const { return fused_; }

 private:
  // non-fused linear layers
  std::vector<LegacyColumnParallelLinear> parallel_linears_;

  // fused linear layer
  LegacyColumnParallelLinear fused_linear_{nullptr};

  // sizes for each split
  std::vector<int64_t> split_sizes_;

  // whether the linear layer is fused
  bool fused_ = false;
};
LLM_MODULE(LegacyFusedColumnParallelLinear);

class FusedColumnParallelLinearImpl : public Module {
 public:
  FusedColumnParallelLinearImpl(int64_t in_features,
                                const std::vector<int64_t>& out_features,
                                const std::vector<std::string>& prefixes,
                                bool bias,
                                bool gather_output,
                                const QuantArgs& quant_args,
                                const ParallelArgs& parallel_args,
                                const torch::TensorOptions& options);

  std::vector<torch::Tensor> forward(torch::Tensor input);

  // load weights from the checkpoint, override this method if necessary
  // returns the number of loaded parameters
  size_t load(const StateDict& state_dict,
              const std::string& name_prefix = std::string()) override;

  // verify whether the weights are loaded, override this method if necessary
  bool verify(const std::string& name_prefix = std::string()) const override;

  // whether the linear layer is fused
  bool fused() const { return fused_; }

 private:
  // non-fused linear layers
  std::vector<LegacyColumnParallelLinear> parallel_linears_;

  // fused linear layer
  LegacyColumnParallelLinear fused_linear_{nullptr};

  // sizes for each split
  std::vector<int64_t> split_sizes_;

  std::vector<std::string> prefixes_;

  // whether the linear layer is fused
  bool fused_ = false;
};
LLM_MODULE(FusedColumnParallelLinear);

}  // namespace llm
