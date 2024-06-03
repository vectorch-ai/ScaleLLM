#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include "linear.h"
#include "model_loader/state_dict.h"
#include "model_parallel/parallel_args.h"
#include "quantization/quant_args.h"

namespace llm {

// a thin wrapper to handle state_dict loading for QKV with
// support of MQA/GQA
class QKVColumnParallelLinearImpl : public torch::nn::Module {
 public:
  QKVColumnParallelLinearImpl(int64_t hidden_size,
                              int64_t n_heads,
                              int64_t n_kv_heads,
                              int64_t head_dim,
                              bool bias,
                              bool gather_output,
                              const QuantArgs& quant_args,
                              const ParallelArgs& parallel_args,
                              const torch::TensorOptions& options);

  torch::Tensor forward(torch::Tensor input) const {
    return parallel_linear_->forward(input);
  }

  // special load_state_dict for fused cases
  void load_state_dict(const StateDict& state_dict,
                       const std::vector<std::string>& prefixes,
                       const std::vector<std::string>& kv_prefixes);

  void verify_loaded_weights(const std::string& prefix = "") const {
    parallel_linear_->verify_loaded_weights(prefix);
  }

 private:
  ColumnParallelLinear parallel_linear_{nullptr};

  // replication ratio of kv heads for MQA/GQA cases
  int64_t kv_replication_ratio_ = 0;

  int64_t n_kv_heads_ = 0;

  int64_t head_dim_ = 0;
};
TORCH_MODULE(QKVColumnParallelLinear);

}  // namespace llm
