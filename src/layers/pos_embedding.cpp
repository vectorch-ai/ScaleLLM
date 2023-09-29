#include "pos_embedding.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <memory>

namespace llm {

namespace {
using torch::indexing::None;
using torch::indexing::Slice;

// [1, 2, 3, 4] => [-2, 1, -4, 3]
inline torch::Tensor rotate_every_two(const torch::Tensor& x) {
  auto x1 = x.index({Slice(), Slice(), Slice(0, None, 2)});
  auto x2 = x.index({Slice(), Slice(), Slice(1, None, 2)});
  return torch::stack({-x2, x1}, /*dim=*/-1).flatten(/*start_dim=*/-2);
}

// apply interleaved rotary positional embedding
inline std::tuple<torch::Tensor, torch::Tensor>
apply_interleaved_rotary_pos_emb(const torch::Tensor& q,
                                 const torch::Tensor& k,
                                 const torch::Tensor& cos,
                                 const torch::Tensor& sin) {
  auto q_embed = (q * cos) + (rotate_every_two(q) * sin);
  auto k_embed = (k * cos) + (rotate_every_two(k) * sin);
  return std::make_tuple(q_embed, k_embed);
}

// [1, 2, 3, 4] => [-3, -4, 1, 2]
inline torch::Tensor rotate_half(const torch::Tensor& x) {
  auto chunks = x.chunk(2, /*dim=*/-1);
  return torch::cat({-chunks[1], chunks[0]}, /*dim=*/-1);
}

// apply rotary positional embedding
inline std::tuple<torch::Tensor, torch::Tensor> apply_rotary_pos_emb(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& cos,
    const torch::Tensor& sin) {
  auto q_embed = (q * cos) + (rotate_half(q) * sin);
  auto k_embed = (k * cos) + (rotate_half(k) * sin);
  return std::make_tuple(q_embed, k_embed);
}

// create right instance based on params
std::shared_ptr<RotaryEmbeddingImpl> create(int64_t rotary_dim,
                                            int64_t max_seq_len,
                                            float scaling_factor,
                                            float rope_theta,
                                            bool interleaved,
                                            torch::ScalarType dtype,
                                            const torch::Device& device) {
  if (interleaved) {
    return std::make_shared<InterleavedRotaryEmbedding>(
        rotary_dim, max_seq_len, scaling_factor, rope_theta, dtype, device);
  }
  return std::make_shared<RotatedRotaryEmbedding>(
      rotary_dim, max_seq_len, scaling_factor, rope_theta, dtype, device);
}
}  // namespace

RotaryEmbedding::RotaryEmbedding(int64_t rotary_dim,
                                 int64_t max_seq_len,
                                 float scaling_factor,
                                 float rope_theta,
                                 bool interleaved,
                                 torch::ScalarType dtype,
                                 const torch::Device& device)
    : ModuleHolder(create(rotary_dim,
                          max_seq_len,
                          scaling_factor,
                          rope_theta,
                          interleaved,
                          dtype,
                          device)) {}

InterleavedRotaryEmbedding::InterleavedRotaryEmbedding(
    int64_t rotary_dim,
    int64_t max_seq_len,
    float scaling_factor,
    float theta,
    torch::ScalarType dtype,
    const torch::Device& device) {
  CHECK(rotary_dim % 2 == 0) << "rotary_dim must be even";
  // Create cos and sin embeddings.
  const auto slice = torch::arange(0, rotary_dim, 2);
  const auto inv_freq = 1.0 / torch::pow(theta, slice / rotary_dim);
  auto t = torch::arange(0, max_seq_len, 1);
  if (scaling_factor != 0) {
    t /= scaling_factor;
  }
  const auto freqs = torch::einsum("i,j->ij", {t, inv_freq});
  // [a, b, c, d] => [a, a, b, b, c, c, d, d]
  auto emd = torch::repeat_interleave(freqs, /*repeats=*/2, /*dim=*/-1);
  emd = emd.to(torch::dtype(dtype).device(device));
  // [max_seq_len, rotary_dim] => [max_seq_len, rotary_dim*2]
  const auto cos_sin = torch::cat({emd.cos(), emd.sin()}, /*dim=*/-1);
  cos_sin_cache_ = register_buffer("cos_sin_cached", cos_sin);
}

// inplace rotary positional embedding
std::tuple<torch::Tensor, torch::Tensor> InterleavedRotaryEmbedding::forward(
    const torch::Tensor& query,     // [num_tokens, n_heads, head_dim]
    const torch::Tensor& key,       // [num_tokens, n_kv_heads, head_dim]
    const torch::Tensor& positions  // [num_tokens]
) const {
  namespace F = torch::nn::functional;
  auto cos_sin = F::embedding(positions, cos_sin_cache_);
  // add a new dimension for n_heads
  cos_sin = cos_sin.unsqueeze(1);
  const auto chunks = cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
  return apply_interleaved_rotary_pos_emb(query, key, chunks[0], chunks[1]);
}

RotatedRotaryEmbedding::RotatedRotaryEmbedding(int64_t rotary_dim,
                                               int64_t max_seq_len,
                                               float scaling_factor,
                                               float theta,
                                               torch::ScalarType dtype,
                                               const torch::Device& device) {
  CHECK(rotary_dim % 2 == 0) << "rotary_dim must be even";
  // Create cos and sin embeddings.
  const auto slice = torch::arange(0, rotary_dim, 2);
  const auto inv_freq = 1.0 / torch::pow(theta, slice / rotary_dim);
  auto t = torch::arange(0, max_seq_len, 1);
  if (scaling_factor != 0) {
    t /= scaling_factor;
  }
  const auto freqs = torch::einsum("i,j->ij", {t, inv_freq});
  // [a, b, c, d] => [a, b, c, d, a, b, c, d]
  auto emd = torch::cat({freqs, freqs}, /*dim=*/-1);
  emd = emd.to(torch::dtype(dtype).device(device));
  const auto cos_sin = torch::cat({emd.cos(), emd.sin()}, /*dim=*/-1);
  cos_sin_cache_ = register_buffer("cos_sin_cached", cos_sin);
}

// inplace rotary positional embedding
std::tuple<torch::Tensor, torch::Tensor> RotatedRotaryEmbedding::forward(
    const torch::Tensor& query,     // [num_tokens, n_heads, head_dim]
    const torch::Tensor& key,       // [num_tokens, n_kv_heads, head_dim]
    const torch::Tensor& positions  // [num_tokens]
) const {
  namespace F = torch::nn::functional;
  auto cos_sin = F::embedding(positions, cos_sin_cache_);
  // add a new dimension for n_heads
  cos_sin = cos_sin.unsqueeze(1);
  const auto chunks = cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
  return apply_rotary_pos_emb(query, key, chunks[0], chunks[1]);
}

}  // namespace llm
