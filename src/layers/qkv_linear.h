#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include "linear.h"
#include "model_loader/state_dict.h"
#include "model_parallel/parallel_args.h"
#include "quantization/quant_args.h"

namespace llm {

// a wrapper to take care of state_dict loading and verification for QKV with
// support of MQA/GQA
class QKVColumnParallelLinearImpl : public ParallelLinearImpl {
 public:
  QKVColumnParallelLinearImpl(int64_t hidden_size,
                              int64_t n_heads,
                              int64_t n_kv_heads,
                              int64_t head_size,
                              bool bias,
                              const ParallelArgs& parallel_args,
                              const torch::TensorOptions& options);

  torch::Tensor forward(torch::Tensor input) const override {
    return parallel_linear_->forward(input);
  }

  void load_state_dict(const StateDict& state_dict) override;

  void verify_loaded_weights(const std::string& prefix = "") const override;

 private:
  std::unique_ptr<ParallelLinearImpl> parallel_linear_;
};

}  // namespace llm
