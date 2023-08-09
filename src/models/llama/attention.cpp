#include "attention.h"

#include <torch/torch.h>

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
  // initialize attention
  attn_ = register_module("attn", SelfAttention());
}

// input: (bsz, seqlen, dim)
// TODO: move freqs_cis and mask to a separate class
torch::Tensor AttentionImpl::forward(torch::Tensor x,
                                     int64_t start_pos,
                                     torch::Tensor mask) {
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));

  const auto bsz = x.size(0);
  const auto seqlen = x.size(1);
  // (bsz, seqlen, dim) x (dim, n_heads * head_dim)
  // (bsz, seqlen, n_heads * head_dim
  auto xq = wq_->forward(x);
  // (bsz, seqlen, n_kv_heads * head_dim)
  auto xk = wk_->forward(x);
  // (bsz, seqlen, n_kv_heads * head_dim)
  auto xv = wv_->forward(x);

  // (bsz, seqlen, n_local_heads, head_dim)
  xq = xq.view({bsz, seqlen, n_local_heads_, head_dim_});
  // (bsz, seqlen, n_local_kv_heads, head_dim)
  xk = xk.view({bsz, seqlen, n_local_kv_heads_, head_dim_});
  // (bsz, seqlen, n_local_kv_heads, head_dim)
  xv = xv.view({bsz, seqlen, n_local_kv_heads_, head_dim_});

  // (bsz, seqlen, n_local_heads, head_dim)
  pos_emb_->forward(xq, xk, start_pos, seqlen);

  // cache k and v (move to a separate function/class)
  // why query can't be cached?
  cache_k_ = cache_k_.to(xq);
  cache_v_ = cache_v_.to(xq);
  using torch::indexing::Slice;
  cache_k_.index_put_({Slice(0, bsz), Slice(start_pos, start_pos + seqlen)},
                      xk);
  cache_v_.index_put_({Slice(0, bsz), Slice(start_pos, start_pos + seqlen)},
                      xv);

  auto keys = cache_k_.index({Slice(0, bsz), Slice(0, start_pos + seqlen)});
  auto values = cache_v_.index({Slice(0, bsz), Slice(0, start_pos + seqlen)});

  if (n_rep_ > 1) {
    // (bs, seqlen, n_local_heads, head_dim)
    // -> (bs, seqlen, n_local_kv_heads*n_rep, head_dim)
    keys = repeat_kv(keys, n_rep_);
    values = repeat_kv(values, n_rep_);
  }

  auto output = attn_->forward(xq, keys, values, mask, scale);
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
