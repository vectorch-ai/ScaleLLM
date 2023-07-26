#include "attention.h"

#include <c10/core/TensorImpl.h>
#include <torch/nn/module.h>
#include <torch/torch.h>

#include "models/linear.h"

namespace llm {

AttentionImpl::AttentionImpl(const ModelArgs& args, int64_t world_size)
    : world_size_(world_size) {
  const int64_t n_kv_heads =
      args.n_kv_heads() == 0 ? args.n_heads() : args.n_kv_heads();
  const int64_t n_local_heads = args.n_heads() / world_size_;
  const int64_t n_local_kv_heads = n_kv_heads / world_size_;
  const int64_t dim = args.dim();
  const int64_t n_heads = args.n_heads();
  const int64_t head_dim = args.dim() / args.n_heads();

  // register submodules
  wq_ = register_module(
      "wq", ColumnParallelLinear(dim, n_heads * head_dim, world_size));
  wk_ = register_module(
      "wk", ColumnParallelLinear(dim, n_kv_heads * head_dim, world_size));
  wv_ = register_module(
      "wv", ColumnParallelLinear(dim, n_kv_heads * head_dim, world_size));
  wo_ = register_module("wo",
                        RowParallelLinear(n_heads * head_dim, dim, world_size));

  // initialize cache
  cache_k_ = torch::zeros(
      {args.max_batch_size(), args.max_seq_len(), n_local_kv_heads, head_dim});
  cache_v_ = torch::zeros(
      {args.max_batch_size(), args.max_seq_len(), n_local_kv_heads, head_dim});
}

torch::Tensor AttentionImpl::forward(torch::Tensor input) {
  return input;
}

// load the weight from the checkpoint
void AttentionImpl::load_state_dict(const StateDict& state_dict) {
  // call each submodule's load_state_dict function
  wq_->load_state_dict(state_dict.select("wq."));
  wk_->load_state_dict(state_dict.select("wk."));
  wv_->load_state_dict(state_dict.select("wv."));
  wo_->load_state_dict(state_dict.select("wo."));
}

}  // namespace llm
