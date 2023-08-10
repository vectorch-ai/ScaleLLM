#include "attention.h"

#include <torch/torch.h>

#include "layers/attention.h"
#include "layers/linear.h"

namespace llm {

static torch::Tensor repeat_kv(const torch::Tensor& x, int64_t n_rep) {
  const auto bs = x.size(0);
  const auto slen = x.size(1);
  const auto n_kv_heads = x.size(2);
  const auto head_dim = x.size(3);
  // (bs, seqlen, n_kv_heads, head_dim)
  // -> (bs, seqlen, n_kv_heads, 1, head_dim)
  // -> (bs, seqlen, n_kv_heads, n_rep, head_dim)
  // -> (bs, seqlen, n_kv_heads * n_rep, head_dim)
  auto x_expanded =
      x.unsqueeze(3).expand({bs, slen, n_kv_heads, n_rep, head_dim});
  return x_expanded.reshape({bs, slen, n_kv_heads * n_rep, head_dim});
}

AttentionImpl::AttentionImpl(const ModelArgs& args, int64_t world_size)
    : world_size_(world_size) {
  if (args.n_kv_heads().has_value()) {
    n_kv_heads_ = args.n_kv_heads().value();
  } else {
    n_kv_heads_ = args.n_heads();
  }
  n_local_heads_ = args.n_heads() / world_size_;
  n_local_kv_heads_ = n_kv_heads_ / world_size_;
  n_rep_ = n_local_heads_ / n_local_kv_heads_;
  head_dim_ = args.dim() / args.n_heads();

  const int64_t dim = args.dim();
  const int64_t n_heads = args.n_heads();

  // register submodules
  wq_ = register_module(
      "wq", ColumnParallelLinear(dim, n_heads * head_dim_, world_size));
  wk_ = register_module(
      "wk", ColumnParallelLinear(dim, n_kv_heads_ * head_dim_, world_size));
  wv_ = register_module(
      "wv", ColumnParallelLinear(dim, n_kv_heads_ * head_dim_, world_size));
  wo_ = register_module(
      "wo", RowParallelLinear(n_heads * head_dim_, dim, world_size));

  // initialize cache (batch_size, seq_len, n_local_kv_heads, head_dim)
  cache_k_ = torch::zeros({args.max_batch_size(),
                           args.max_seq_len(),
                           n_local_kv_heads_,
                           head_dim_});
  cache_v_ = torch::zeros({args.max_batch_size(),
                           args.max_seq_len(),
                           n_local_kv_heads_,
                           head_dim_});

  // initialize positional embedding
  pos_emb_ =
      register_module("pos_emb",
                      RotaryPositionalEmbedding(args.dim() / args.n_heads(),
                                                args.max_seq_len()));
}

// x : [num_tokens, dim]
// positions : [num_tokens]
torch::Tensor AttentionImpl::forward(torch::Tensor x,
                                     torch::Tensor positions,
                                     const std::vector<int64_t>& cu_seq_lens) {
  const auto num_tokens = x.size(0);
  // (num_tokens, dim) x (dim, n_heads * head_dim)
  // => (num_tokens, n_heads * head_dim)
  auto xq = wq_->forward(x);
  auto xk = wk_->forward(x);
  auto xv = wv_->forward(x);

  // (num_tokens, n_local_heads, head_dim)
  xq = xq.view({num_tokens, n_local_heads_, head_dim_});
  xk = xk.view({num_tokens, n_local_kv_heads_, head_dim_});
  xv = xv.view({num_tokens, n_local_kv_heads_, head_dim_});

  // (num_tokens, n_local_heads, head_dim)
  pos_emb_->forward(xq, xk, positions);

  // TODO: add blocked cache support

  auto output =
      attention::varlen_masked_self_attention(xq, xk, xv, cu_seq_lens);
  output = output.contiguous().view({num_tokens, -1});
  return wo_->forward(output);
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
