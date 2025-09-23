#include "fused_linear.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include "linear.h"
#include "model_loader/state_dict.h"
#include "model_parallel/parallel_args.h"
#include "quantization/quant_args.h"

namespace llm {

FusedColumnParallelLinearImpl::FusedColumnParallelLinearImpl(
    int64_t in_features,
    const std::vector<int64_t>& out_features_vec,
    bool bias,
    bool gather_output,
    const QuantArgs& quant_args,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options) {
  // check if the linear layers can be fused
  fused_ = quant_args.can_be_fused();
  if (fused_) {
    // fused linear layer
    const int64_t out_features = std::accumulate(
        out_features_vec.begin(), out_features_vec.end(), int64_t(0));
    fused_linear_ = LegacyColumnParallelLinear(in_features,
                                               out_features,
                                               bias,
                                               gather_output,
                                               quant_args,
                                               parallel_args,
                                               options);
    // calculate split sizes
    split_sizes_.reserve(out_features_vec.size());
    const auto world_size = parallel_args.world_size();
    for (const auto& out_features : out_features_vec) {
      CHECK(out_features % world_size == 0)
          << "out_features " << out_features << " not divisible by world_size "
          << world_size;
      split_sizes_.push_back(out_features / world_size);
    }
  } else {
    // non-fused linear layers
    parallel_linears_.reserve(out_features_vec.size());
    for (const auto& out_features : out_features_vec) {
      parallel_linears_.emplace_back(in_features,
                                     out_features,
                                     bias,
                                     gather_output,
                                     quant_args,
                                     parallel_args,
                                     options);
    }
  }
}

std::vector<torch::Tensor> FusedColumnParallelLinearImpl::forward(
    torch::Tensor input) {
  if (fused_) {
    auto fused_output = fused_linear_->forward(input);
    return fused_output.split(split_sizes_, /*dim=*/1);
  }

  // otherwise, use the non-fused linear layers
  std::vector<torch::Tensor> outputs;
  outputs.reserve(parallel_linears_.size());
  for (auto& parallel_linear : parallel_linears_) {
    auto output = parallel_linear->forward(input);
    outputs.push_back(output);
  }
  return outputs;
}

void FusedColumnParallelLinearImpl::load_state_dict(
    const StateDict& state_dict,
    const std::vector<std::string>& prefixes) {
  if (fused_) {
    fused_linear_->load_state_dict(state_dict, prefixes);
  } else {
    CHECK_EQ(parallel_linears_.size(), prefixes.size());
    for (size_t i = 0; i < parallel_linears_.size(); ++i) {
      parallel_linears_[i]->load_state_dict(state_dict.select(prefixes[i]));
    }
  }
}

void FusedColumnParallelLinearImpl::verify_loaded_weights(
    const std::string& prefix) const {
  if (fused_) {
    fused_linear_->verify_loaded_weights(prefix);
  } else {
    for (const auto& parallel_linear : parallel_linears_) {
      parallel_linear->verify_loaded_weights(prefix);
    }
  }
}

}  // namespace llm
