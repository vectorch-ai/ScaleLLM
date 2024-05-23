#include "fused_moe.h"

#include <torch/torch.h>

#include "kernels/fused_moe_kernels.h"
#include "models/model_args.h"
#include "models/parameters.h"
namespace llm{
FusedMoeLayerImpl::FusedMoeLayerImpl(bool renormalize,
                                     bool inplace,
                                     const ModelArgs& args,
                                     const QuantArgs& quant_args,
                                     const ParallelArgs& parallel_args,
                                     const torch::TensorOptions& options)
    : parallel_args_(parallel_args),
      renormalize_(renormalize),
      inplace_(inplace) {
  topk_ = args.n_experts_per_tok();
  num_total_experts_ = args.n_local_experts();
  intermediate_size_ = args.intermediate_size();
  hidden_size_ = args.hidden_size();

  w13_ = register_parameter(
      "weight",
      torch::empty({num_total_experts_, 2 * intermediate_size_, hidden_size_},
                   options),
      /*required_grad*/ false);
  w2_ = register_parameter(
      "weight",
      torch::empty({num_total_experts_, hidden_size_, intermediate_size_},
                   options),
      /*required_grad*/ false);
}

torch::Tensor FusedMoeLayerImpl::forward(
    torch::Tensor hidden_states,  // [hidden_size,hidden_dim]
    torch::Tensor gating_output   // [n_tokens,n_expert]
) {
  // ========  fused_topk: topk(softmax(gating_output))============
  // match the number of tokens
  DCHECK_EQ(hidden_states.sizes()[0], gating_output.sizes()[0]);
  auto router_weight = torch::softmax(gating_output, -1, torch::kFloat32);
  auto [topk_weights, topk_indices] =
      torch::topk(router_weight, topk_, -1);  // [n_tokens,n_topk]

  if (renormalize_) {
    topk_weights = topk_weights / topk_weights.sum(-1, true);
  }

  // ================   fused_expert =================
  // be sure that hidden_states/w13/w2 are contiguous
  DCHECK_EQ(hidden_states.is_contiguous(), true);
  DCHECK_EQ(w13_.is_contiguous(), true);
  DCHECK_EQ(w2_.is_contiguous(), true);
  return kernel::apply_fused_moe(hidden_states,w13_,w2_,topk_weights,topk_indices,inplace_);
}

void FusedMoeLayerImpl::load_state_dict(const StateDict& state_dict) {
  // prefix:model.layers.0.block_sparse_moe.experts.
  auto shard_size = intermediate_size_;
  auto world_size = shard_size * state_dict.num_shards();
  if (w2_.defined()) {
    for (int i = 0; i < num_total_experts_; i++) {
      auto w2 = state_dict.select(std::to_string(i) + ".w2.")
                    .get_sharded_tensor("weight",
                                        /*dim*/ 1,
                                        /*rank*/ parallel_args_.rank(),
                                        /*world_size*/ world_size);
      w2_.slice(0, i).copy_(w2);
    }
    is_loaded_w2_ = true;
  }
  if (w13_.defined()) {
    for (int i = 0; i < num_total_experts_; i++) {
      auto w1 = state_dict.select(std::to_string(i) + ".w1.")
                    .get_sharded_tensor("weight",
                                        /*dim*/ 0,
                                        /*rank*/ parallel_args_.rank(),
                                        /*world_size*/ world_size);
      w13_.slice(0, i).slice(1, 0, shard_size).copy_(w1);
      auto w3 = state_dict.select(std::to_string(i) + ".w3.")
                    .get_sharded_tensor("weight",
                                        /*dim*/ 0,
                                        /*rank*/ parallel_args_.rank(),
                                        /*world_size*/ world_size);
      w13_.slice(0, i).slice(1, shard_size, 2 * shard_size).copy_(w3);
    }
    is_loaded_w13_ = true;
  }
}

} //namespace llm