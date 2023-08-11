#include "pos_embedding.h"

#include <glog/logging.h>
#include <torch/torch.h>

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

// returns a tensor where the last dimension of the original tensor is split
// into two dimensions shape from [..., n] to [..., -1, 2]
torch::Tensor split_tensor_by_last_dim(const torch::Tensor& x) {
  auto shape = x.sizes().vec();
  shape.back() = -1;
  shape.push_back(2);
  return x.reshape(shape);
}

// xq: (num_tokens, n_local_heads, head_dim)
void apply_rotary_emb(torch::Tensor& xq,
                      torch::Tensor& xk,
                      torch::Tensor freqs_cis) {
  // (num_tokens, n_local_heads, head_dim/2, 2)
  //  -> (num_tokens, n_local_heads, head_dim/2)
  auto xq_complex =
      torch::view_as_complex(split_tensor_by_last_dim(xq.to(torch::kFloat32)));
  auto xk_complex =
      torch::view_as_complex(split_tensor_by_last_dim(xk.to(torch::kFloat32)));

  // reshape for broadcast at n_heads dim => (num_tokens, 1 (n_heads), head_dim/2)
  freqs_cis = freqs_cis.unsqueeze(1);
  // -> (num_tokens, n_heads, head_dim)
  auto xq_out = torch::view_as_real(xq_complex * freqs_cis).flatten(2);
  auto xk_out = torch::view_as_real(xk_complex * freqs_cis).flatten(2);
  xq = xq_out.type_as(xq);
  xk = xk_out.type_as(xk);
}
}  // namespace

namespace llm {

RotaryPositionalEmbeddingImpl::RotaryPositionalEmbeddingImpl(
    int64_t rotary_dim,
    int64_t max_seq_len) {
  // calculate freqs_cis
  freqs_cis_ = precompute_freqs_cis(rotary_dim, max_seq_len * 2);
}

void RotaryPositionalEmbeddingImpl::forward(
    torch::Tensor& query,    // [num_tokens, n_heads, head_dim]
    torch::Tensor& key,      // [num_tokens, n_kv_heads, head_dim]
    torch::Tensor positions  // [num_tokens]
) const {
  namespace F = torch::nn::functional;
  auto freqs_cis = F::embedding(positions, freqs_cis_);
  apply_rotary_emb(query, key, freqs_cis);
}

}  // namespace llm
