#pragma once
#include <torch/torch.h>

#include "model_loader/state_dict.h"
#include "model_parallel/parallel_args.h"
#include "models/model_args.h"
#include "quantization/quant_args.h"
namespace llm {

// ======= a Mixture of Experts (MoE) layer using two sets of weights, w1 and
// w2, and top-k gating mechanis ======== Main parameters are below:
// - w1 (torch.Tensor): The first set of expert weights.
//       [n_expert,2*intermediate_size,hidden_size]
// - w2 (torch.Tensor): The second set of expert weights.
//       [n_expert,hidden_size,intermediate_size]
// - topk (int): The number of top-k experts to select.
// - renormalize (bool): If True, renormalize the top-k weights to sum to 1.
// - inplace (bool): If True, perform the operation in-place.
class FusedMoeLayerImpl : public torch::nn::Module {
 public:
  FusedMoeLayerImpl(bool renormalize,
                    bool inplace,
                    const ModelArgs& args,
                    const QuantArgs& quant_args,
                    const ParallelArgs& parallel_args,
                    const torch::TensorOptions& options);
  torch::Tensor forward(torch::Tensor hidden_states, torch::Tensor gating_out);
  void load_state_dict(const StateDict& state_dict);
  void verify_loaded_weights(const std::string& prefix = "") const {
    CHECK(is_loaded_w13_) << "weight is not loaded for" << prefix + "weight13";
    CHECK(is_loaded_w2_) << "weight is not loaded for" << prefix + "weight2";
  };

 private:
  ParallelArgs parallel_args_;
  // for gate mechanism
  int topk_ = 0;
  int num_total_experts_;
  int intermediate_size_;
  int hidden_size_;
  bool renormalize_;
  bool inplace_;
  // for expert mechanism
  torch::Tensor w13_{nullptr};
  torch::Tensor w2_{nullptr};

  // whether the weight is loaded
  bool is_loaded_w13_ = false;
  bool is_loaded_w2_ = false;
};
TORCH_MODULE(FusedMoeLayer);
}  // namespace llm