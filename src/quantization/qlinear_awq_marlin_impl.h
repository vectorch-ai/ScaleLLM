#pragma once

#include <ATen/core/TensorBody.h>
#include <torch/torch.h>

#include "layers/linear/linear.h"
#include "layers/linear/weight_utils.h"
#include "model_loader/state_dict.h"
#include "model_parallel/parallel_args.h"

namespace llm {

class ColumnParallelQLinearAWQMarlinImpl : public ParallelLinearImpl {
 public:
  ColumnParallelQLinearAWQMarlinImpl(int64_t in_features,
                                     int64_t out_features,
                                     bool bias,
                                     const QuantArgs& quant_args,
                                     bool gather_output,
                                     const ParallelArgs& parallel_args,
                                     const torch::TensorOptions& options);

  // verify if the weight is loaded correctly
  void verify_loaded_weights(const std::string& prefix = "") const override;

  torch::Tensor forward(torch::Tensor input) override;

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) override;

  // special load_state_dict for fused cases
  void load_state_dict(const StateDict& state_dict,
                       const std::vector<std::string>& prefixes) override;

 private:
  // parameter members, must be registered
  DEFINE_FUSED_WEIGHT(qweight);
  DEFINE_FUSED_WEIGHT(qzeros);
  DEFINE_FUSED_WEIGHT(scales);
  DEFINE_FUSED_WEIGHT(bias);

  // buffers
  torch::Tensor workspace_;
  torch::Tensor g_idx_;
  torch::Tensor perm_;

  // quantization parameters
  int64_t bits_ = 0;
  bool weight_repacked_ = false;

  // whether to gather the output
  bool gather_output_ = false;

  // parallel args
  ParallelArgs parallel_args_;
};

class RowParallelQLinearAWQMarlinImpl : public ParallelLinearImpl {
 public:
  RowParallelQLinearAWQMarlinImpl(int64_t in_features,
                                  int64_t out_features,
                                  bool bias,
                                  const QuantArgs& quant_args,
                                  bool input_is_parallelized,
                                  const ParallelArgs& parallel_args,
                                  const torch::TensorOptions& options);

  torch::Tensor forward(torch::Tensor input) override;

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) override;

  // whether the weight is loaded
  void verify_loaded_weights(const std::string& prefix = "") const override;

 private:
  // parameter members, must be registered
  DEFINE_WEIGHT(qweight);
  DEFINE_WEIGHT(qzeros);
  DEFINE_WEIGHT(scales);
  DEFINE_WEIGHT(bias);

  // buffers
  torch::Tensor workspace_;
  torch::Tensor g_idx_;
  torch::Tensor perm_;

  // quantization parameters
  int64_t bits_ = 0;
  bool weight_repacked_ = false;

  // whether the input is already parallelized
  bool input_is_parallelized_;

  // parallel args
  ParallelArgs parallel_args_;
};
}  // namespace llm
