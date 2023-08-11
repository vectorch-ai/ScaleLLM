#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include <tuple>

namespace llm {

namespace details {
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

}  // namespace details

inline constexpr float kDefaultTheta = 10000.0f;

// Two types of rotary positional embeddings:
// 1> Interleaved rotation style: rotates pairs of even and odd dimensions
// (used in GPT-J and LLama).
class InterleavedRotaryEmbeddingImpl : public torch::nn::Module {
 public:
  InterleavedRotaryEmbeddingImpl(int64_t rotary_dim,
                                 int64_t max_seq_len,
                                 float theta = kDefaultTheta) {
    // Create cos and sin embeddings.
    const auto slice = torch::arange(0, rotary_dim, 2).to(torch::kFloat32);
    const auto inv_freq = 1.0 / torch::pow(theta, slice / rotary_dim);
    const auto t = torch::arange(0, max_seq_len, 1).to(torch::kFloat32);
    const auto freqs =
        torch::einsum("i,j->ij", {t, inv_freq.to(torch::kFloat32)});
    // [a, b, c, d] => [a, a, b, b, c, c, d, d]
    const auto emd = torch::repeat_interleave(freqs, /*repeats=*/2, /*dim=*/-1);
    // [max_seq_len, rotary_dim] => [max_seq_len, rotary_dim*2]
    const auto cos_sin = torch::cat({emd.cos(), emd.sin()}, /*dim=*/-1);
    cos_sin_cache_ = register_buffer("cos_sin_cached", cos_sin);
  }

  // inplace rotary positional embedding
  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& query,     // [num_tokens, n_heads, head_dim]
      const torch::Tensor& key,       // [num_tokens, n_kv_heads, head_dim]
      const torch::Tensor& positions  // [num_tokens]
  ) const {
    namespace F = torch::nn::functional;
    auto cos_sin = F::embedding(positions, cos_sin_cache_);
    // add a new dimension for n_heads
    cos_sin = cos_sin.unsqueeze(1);
    const auto chunks = cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
    return details::apply_interleaved_rotary_pos_emb(
        query, key, chunks[0], chunks[1]);
  }

 private:
  torch::Tensor cos_sin_cache_;
};
TORCH_MODULE(InterleavedRotaryEmbedding);

// 2> Half-half rotation style of rotary positional embedding (as seen in
// GPT-Neo).
class RotaryEmbeddingImpl : public torch::nn::Module {
 public:
  RotaryEmbeddingImpl(int64_t rotary_dim,
                      int64_t max_seq_len,
                      float theta = kDefaultTheta) {
    // Create cos and sin embeddings.
    const auto slice = torch::arange(0, rotary_dim, 2).to(torch::kFloat32);
    const auto inv_freq = 1.0 / torch::pow(theta, slice / rotary_dim);
    const auto t = torch::arange(0, max_seq_len, 1).to(torch::kFloat32);
    const auto freqs =
        torch::einsum("i,j->ij", {t, inv_freq.to(torch::kFloat32)});
    // [a, b, c, d] => [a, b, c, d, a, b, c, d]
    const auto emd = torch::cat({freqs, freqs}, /*dim=*/-1);
    const auto cos_sin = torch::cat({emd.cos(), emd.sin()}, /*dim=*/-1);
    cos_sin_cache_ = register_buffer("cos_sin_cached", cos_sin);
  }

  // inplace rotary positional embedding
  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& query,     // [num_tokens, n_heads, head_dim]
      const torch::Tensor& key,       // [num_tokens, n_kv_heads, head_dim]
      const torch::Tensor& positions  // [num_tokens]
  ) const {
    namespace F = torch::nn::functional;
    auto cos_sin = F::embedding(positions, cos_sin_cache_);
    // add a new dimension for n_heads
    cos_sin = cos_sin.unsqueeze(1);
    const auto chunks = cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
    return details::apply_rotary_pos_emb(query, key, chunks[0], chunks[1]);
  }

 private:
  torch::Tensor cos_sin_cache_;
};
TORCH_MODULE(RotaryEmbedding);

}  // namespace llm
