#pragma once
#include <torch/torch.h>

namespace llm {

// ======= a Mixture of Experts (MoE) layer using two sets of weights, w1 and
// w2, and top-k gating mechanis ======== Main parameters are below:
// - w1 (torch.Tensor): The first set of expert weights.[n_expert,,hidden_size]
// - w2 (torch.Tensor): The second set of expert weights.
// - topk (int): The number of top-k experts to select.
// - renormalize (bool): If True, renormalize the top-k weights to sum to 1.
// - inplace (bool): If True, perform the operation in-place.
class FusedMoeLayerImpl : public torch::nn::Module {
 public:
  FusedMoeLayerImpl(int topk,
                    bool renormalize,
                    bool inplace,
                    const QuantArgs& quant_args,
                    const torch::TensorOptions& options) override;
  torch::Tensor forward(torch::Tensor hidden_states,
                        torch::Tensor gating_out) override;
  void load_state_dict(const StateDict& state_dict) override;
  void verify_loaded_weights(const std::string& prefix = "") const override;

 private:
  // for gate mechanism
  int topk_ = 0;
  bool renormalize_;
  bool inplace_ = false;
  // for expert mechanism
  torch::Tensor w13_{nullptr};
  torch::Tensor w2_{nullptr};
  
  // whether the weight is loaded
  bool is_loaded_w13 = false;
  bool is_loaded_w2 =false;
} TORCH_MODULE(FusedMoeLayer);
}  // namespace llm