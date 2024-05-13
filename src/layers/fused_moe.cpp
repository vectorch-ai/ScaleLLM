#include <torch/torch.h>
#include "fused_moe.h"
#include "kernels/fused_moe_kernels.h"

namespace llm{
FusedMoeLayerImpl::FusedMoeLayerImpl(int topk,
                                     bool renormalize,
                                     bool inplace,
                                     const QuantArgs& quant_args,
                                     const torch::TensorOptions& options)
    : topk_(topk), 
    renormalize_(renormalize), 
    inplace_(inplace) {
  w13_ = register_parameter(
      "weight", torch::empty({}, options), /*required_grad*/ false);
  w2_ = register_parameter(
      "weight", torch::empty({}, options), /*required_grad*/ false);
}

torch::Tensor FusedMoeLayerImpl::forward(
    torch::Tensor hidden_states,  // [hidden_size,hidden_dim]
    torch::Tensor gating_out      // [n_tokens,n_expert]
) {
  // match the number of tokens
  DCHECK_EQ(hidden_states.sizes()[0], gating_output.sizes()[0]);
  // match the number of hidden_size
  DCHECK_EQ(hidden_states.sizes()[1], w13_.sizes()[2]);
  // match the number of experts
  DCHECK_EQ(gating_output.sizes()[1], w13_.sizes()[0]);
  // be sure that hidden_states/w1/w2 are contiguous
  DCHECK_EQ(hidden_states.is_contiguous(), true);

  auto M = hidden_states.sizes()[0];
  auto E = w1.sizes()[0];
  auto N = w1.sizes()[1];

  // ========  fused_topk: topk(softmax(gating_output))============
  auto router_weight = torch::softmax(gating_output, -1, torch::kFloat32);
  auto [topk_weights, topk_indices] =
      torch::topk(router_weight, topk, -1);  // [n_tokens,n_topk]

  if (renormalize_) {
    topk_weights = topk_weights / topk_weights.sum(-1, true);
  }

  // ================   fused_expert =================
  return kernel::apply_fused_moe(hidden_states,w13_,w2_,topk_weights,topk_indices,inplace_);
}

void FusedMoeLayerImpl::load_state_dict(const StateDict& state_dict) {
  DCHECK_EQ(w13_.is_contiguous(), true);
  DCHECK_EQ(w2_.is_contiguous(), true);
}
void FusedMoeLayerImpl::verify_loaded_weights( const std::string& prefix = "") const {
  CHECK(is_loaded_w13) << "weight is not loaded for" << prefix + "weight13";
  CHECK(is_loaded_w2) << "weight is not loaded for"<<prefix+"weight2";
}
} //namespace llm