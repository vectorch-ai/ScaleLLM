#include "pos_embedding.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include "models/model_args.h"

namespace {
torch::Tensor precompute_freqs_cis(int64_t dim,
                                   int64_t max_seq_len,
                                   float theta = 10000.0f) {
  auto range = torch::arange(0, dim, 2);
  auto slice =
      range.slice(/*dim=*/0, /*start=*/0, /*end=*/dim / 2).to(torch::kFloat32);
  auto freqs = 1.0 / torch::pow(theta, slice / dim);
  auto t = torch::arange(0, max_seq_len, 1).to(torch::kFloat32);
  freqs = torch::outer(t, freqs).to(torch::kFloat32);
  return torch::polar(torch::ones_like(freqs), freqs);
}

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
torch::Tensor split_tensor_by_last_dim(const torch::Tensor& x) {
  auto shape = x.sizes().vec();
  shape.back() = -1;
  shape.push_back(2);
  return x.reshape(shape);
}

// xq: (bsz, seqlen, n_local_heads, head_dim)
void apply_rotary_emb(torch::Tensor& xq,
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
}  // namespace

namespace llm {

RotaryPositionalEmbeddingImpl::RotaryPositionalEmbeddingImpl(
    const ModelArgs& args) {
  // calculate freqs_cis
  freqs_cis_ =
      precompute_freqs_cis(args.dim() / args.n_heads(), args.max_seq_len() * 2);
}

void RotaryPositionalEmbeddingImpl::forward(torch::Tensor& query,
                                            torch::Tensor& key,
                                            int64_t start_pos,
                                            int64_t seq_len) const {
  const auto freqs_cis =
      freqs_cis_.slice(/*dim=*/0, start_pos, start_pos + seq_len);
  apply_rotary_emb(query, key, freqs_cis);
}

}  // namespace llm
