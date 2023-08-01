#include "attention.h"

#include <c10/core/TensorImpl.h>
#include <torch/nn/module.h>
#include <torch/torch.h>

#include "layers/layers.h"

namespace llm {
// x (bsz, seqlen, n_local_heads, head_dim/2)
torch::Tensor reshape_for_broadcast(const torch::Tensor& freqs_cis,
                                    const torch::Tensor& x) {
  const int64_t ndim = x.dim();
  std::vector<int64_t> shape(ndim, 1);
  shape[1] = x.size(1);
  shape[ndim - 1] = x.size(ndim - 1);
  // (seqlen, dim/2) -> (1, seqlen, 1, head_dim/2)
  return freqs_cis.view(shape);
}

// returns a tensor where the last dimension of the original tensor is split
// into two dimensions shape from [..., n] to [..., -1, 2]
static torch::Tensor split_tensor_by_last_dim(const torch::Tensor& x) {
  auto shape = x.sizes().vec();
  shape.back() = -1;
  shape.push_back(2);
  return x.reshape(shape);
}

// xq: (bsz, seqlen, n_local_heads, head_dim)
static void apply_rotary_emb(torch::Tensor& xq,
                             torch::Tensor& xk,
                             torch::Tensor freqs_cis) {
  // (bsz, seqlen, n_local_heads, head_dim/2, 2)
  //  -> (bsz, seqlen, n_local_heads, head_dim/2)
  auto xq_complex =
      torch::view_as_complex(split_tensor_by_last_dim(xq.to(torch::kFloat32)));
  auto xk_complex =
      torch::view_as_complex(split_tensor_by_last_dim(xk.to(torch::kFloat32)));

  // (1, seqlen, 1, head_dim/2)
  freqs_cis = reshape_for_broadcast(freqs_cis, xq_complex);
  // (bsz, seqlen, n_local_heads, head_dim/2)
  // -> (bsz, seqlen, n_local_heads, head_dim/2, 2)
  // -> (bsz, seqlen, n_local_heads, head_dim)
  auto xq_out = torch::view_as_real(xq_complex * freqs_cis).flatten(3);
  auto xk_out = torch::view_as_real(xk_complex * freqs_cis).flatten(3);

  xq = xq_out.type_as(xq);
  xk = xk_out.type_as(xk);
}

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
}

// input: (bsz, seqlen, dim)
// TODO: move freqs_cis and mask to a separate class
torch::Tensor AttentionImpl::forward(torch::Tensor x,
                                     int64_t start_pos,
                                     torch::Tensor freqs_cis,
                                     torch::Tensor mask) {
  const double sqrt_head_dim = std::sqrt(static_cast<double>(head_dim_));

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
  apply_rotary_emb(xq, xk, freqs_cis);

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

  // (bs, n_local_heads, seqlen, head_dim)
  xq = xq.transpose(1, 2);
  keys = keys.transpose(1, 2);
  values = values.transpose(1, 2);

  // [bs, n_local_heads, seqlen, head_dim] x [bs, n_local_heads, head_dim,
  // cache_len + seqlen]
  // => [bs, n_local_heads, seqlen, cache_len + seqlen]
  auto scores = torch::matmul(xq, keys.transpose(2, 3)) / sqrt_head_dim;
  if (mask.defined()) {
    // (bs, n_local_heads, seqlen, cache_len + seqlen)
    scores += mask;
  }
  // (bs, n_local_heads, seqlen, cache_len + seqlen)
  scores = torch::softmax(scores.to(torch::kFloat), -1).type_as(xq);
  // (bs, n_local_heads, seqlen, cache_len + seqlen) x [bs, n_local_heads,
  // cache_len + seqlen, head_dim]
  // => [bs, n_local_heads, seqlen, head_dim]
  auto output = torch::matmul(scores, values);
  // (bs, seqlen, dim = n_local_heads X head_dim)
  output = output.transpose(1, 2).contiguous().view({bsz, seqlen, -1});
  // (bs, seqlen, n_local_heads * head_dim) X (n_heads * head_dim_, dim)
  // -> (bs, seqlen, dim)
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
