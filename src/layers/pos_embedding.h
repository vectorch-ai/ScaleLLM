#pragma once

#include <c10/core/Device.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <tuple>

namespace llm {
inline constexpr float kDefaultTheta = 10000.0f;

// an interface for rotary positional embedding.
// all rotary positional embedding classes should inherit from this class and
// implement the forward function.
class RotaryEmbeddingImpl : public torch::nn::Module {
 public:
  ~RotaryEmbeddingImpl() override = default;

  // returns a tuple of query and key embeddings with the same shape as the
  // input query and key.
  virtual std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& query,     // [num_tokens, n_heads, head_dim]
      const torch::Tensor& key,       // [num_tokens, n_kv_heads, head_dim]
      const torch::Tensor& positions  // [num_tokens]
  ) const = 0;
};

// RotaryEmbedding is a wrapper class that chooses the right rotary positional
// embedding implementation based on the args.
// Similar to TORCH_MODULE(RotaryEmbedding) except of the explicit constructor.
class RotaryEmbedding : public torch::nn::ModuleHolder<RotaryEmbeddingImpl> {
 public:
  using torch::nn::ModuleHolder<RotaryEmbeddingImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = RotaryEmbeddingImpl;

  // construct a rotary positional embedding.
  // chose right implementation based on the args.
  RotaryEmbedding(int64_t rotary_dim,
                  int64_t max_seq_len,
                  float scaling_factor,
                  bool interleaved,
                  const torch::Device& device);
};

// ============= Rotary positional embedding implementations =============
// Two types of rotary positional embeddings:
// 1> Interleaved rotation style: rotates pairs of even and odd dimensions
// (used in GPT-J and LLama).
// 2> Half-half rotation style of rotary positional embedding (as seen in
// GPT-Neo).
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
class InterleavedRotaryEmbedding : public RotaryEmbeddingImpl {
 public:
  InterleavedRotaryEmbedding(int64_t rotary_dim,
                             int64_t max_seq_len,
                             float scaling_factor,
                             const torch::Device& device,
                             float theta = kDefaultTheta) {
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
    emd = emd.to(device);
    // [max_seq_len, rotary_dim] => [max_seq_len, rotary_dim*2]
    const auto cos_sin = torch::cat({emd.cos(), emd.sin()}, /*dim=*/-1);
    cos_sin_cache_ = register_buffer("cos_sin_cached", cos_sin);
  }

  // inplace rotary positional embedding
  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& query,     // [num_tokens, n_heads, head_dim]
      const torch::Tensor& key,       // [num_tokens, n_kv_heads, head_dim]
      const torch::Tensor& positions  // [num_tokens]
  ) const override {
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

class RotatedRotaryEmbedding : public RotaryEmbeddingImpl {
 public:
  RotatedRotaryEmbedding(int64_t rotary_dim,
                         int64_t max_seq_len,
                         float scaling_factor,
                         const torch::Device& device,
                         float theta = kDefaultTheta) {
    // Create cos and sin embeddings.
    const auto slice = torch::arange(0, rotary_dim, 2);
    const auto inv_freq = 1.0 / torch::pow(theta, slice / rotary_dim);
    auto t = torch::arange(0, max_seq_len, 1);
    if (scaling_factor != 0) {
      t /= scaling_factor;
    }
    const auto freqs = torch::einsum("i,j->ij", {t, inv_freq});
    // [a, b, c, d] => [a, b, c, d, a, b, c, d]
    const auto emd = torch::cat({freqs, freqs}, /*dim=*/-1);
    emd.to(device);
    const auto cos_sin = torch::cat({emd.cos(), emd.sin()}, /*dim=*/-1);
    cos_sin_cache_ = register_buffer("cos_sin_cached", cos_sin);
  }

  // inplace rotary positional embedding
  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& query,     // [num_tokens, n_heads, head_dim]
      const torch::Tensor& key,       // [num_tokens, n_kv_heads, head_dim]
      const torch::Tensor& positions  // [num_tokens]
  ) const override {
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

}  // namespace llm
