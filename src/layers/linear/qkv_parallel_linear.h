#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include "layers/module/module.h"
#include "layers/module/module_holder.h"
#include "layers/quantization/quant_args.h"
#include "model_loader/state_dict.h"
#include "model_parallel/parallel_args.h"
#include "multi_parallel_linear.h"

namespace llm {

// a thin wrapper to handle state_dict loading for QKV with
// support of MQA/GQA
class QKVColumnParallelLinearImpl : public Module {
 public:
  QKVColumnParallelLinearImpl(int64_t hidden_size,
                              int64_t n_heads,
                              int64_t n_kv_heads,
                              int64_t head_dim,
                              const std::vector<std::string>& prefixes,
                              bool bias,
                              bool gather_output,
                              const QuantArgs& quant_args,
                              const ParallelArgs& parallel_args,
                              const torch::TensorOptions& options);

  // returns (query, key, value)
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
      torch::Tensor input) {
    const auto qkv = parallel_linear_->forward(input);
    return {qkv[0], qkv[1], qkv[2]};
  }

 private:
  // registered modules
  MultiColumnParallelLinear parallel_linear_{nullptr};
};
LLM_MODULE(QKVColumnParallelLinear);

}  // namespace llm
