#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

// #include "linear.h"
#include "layers/quantization/quant_args.h"
#include "model_parallel/parallel_args.h"
#include "module/module.h"
#include "module/module_holder.h"
#include "parallel_linear.h"

namespace llm {

class MultiParallelLinearImpl : public Module {
 public:
  ~MultiParallelLinearImpl() override = default;

  virtual std::vector<torch::Tensor> forward(torch::Tensor input) = 0;
};
LLM_MODULE(MultiParallelLinear);

// Fused linear layer with column parallelism.
class FusedColumnParallelLinearImpl : public MultiParallelLinearImpl {
 public:
  FusedColumnParallelLinearImpl(int64_t in_features,
                                const std::vector<int64_t>& out_features,
                                const std::vector<std::string>& prefixes,
                                bool bias,
                                bool gather_output,
                                const ParallelArgs& parallel_args,
                                const torch::TensorOptions& options);

  std::vector<torch::Tensor> forward(torch::Tensor input) override;

  // return the weight (for testing)
  torch::Tensor weight() const { return weight_; }

 private:
  // parameter members, must be registered
  // we allocate the transpose since linear performs XA^T.
  // A^T: [out_features_per_partition, in_features]
  torch::Tensor weight_;
  torch::Tensor bias_;

  std::vector<int64_t> split_sizes_;

  // whether to gather the output
  bool gather_output_;

  // parallel args
  ParallelArgs parallel_args_;
};
LLM_MODULE(FusedColumnParallelLinear);

class GroupedColumnParallelLinearImpl : public MultiParallelLinearImpl {
 public:
  GroupedColumnParallelLinearImpl(int64_t in_features,
                                  const std::vector<int64_t>& out_features,
                                  const std::vector<std::string>& prefixes,
                                  bool bias,
                                  bool gather_output,
                                  const ParallelArgs& parallel_args,
                                  const torch::TensorOptions& options);

  std::vector<torch::Tensor> forward(torch::Tensor input) override;

 private:
  // parameter members, must be registered
  std::vector<std::shared_ptr<ColumnParallelLinearImpl>> parallel_linears_;
};
LLM_MODULE(GroupedColumnParallelLinear);

class MultiColumnParallelLinearImpl : public Module {
 public:
  MultiColumnParallelLinearImpl(int64_t in_features,
                                const std::vector<int64_t>& out_features,
                                const std::vector<std::string>& prefixes,
                                bool bias,
                                bool gather_output,
                                const QuantArgs& quant_args,
                                const ParallelArgs& parallel_args,
                                const torch::TensorOptions& options);

  std::vector<torch::Tensor> forward(torch::Tensor input);

 private:
  MultiParallelLinear linear_{nullptr};
};
LLM_MODULE(MultiColumnParallelLinear);

}  // namespace llm
