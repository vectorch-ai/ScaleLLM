#pragma once

#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <torch/torch.h>

#include <tuple>

namespace llm {
namespace detail {
torch::Tensor compute_freqs(int64_t max_position_embeddings,
                            int64_t rotary_dim,
                            float scaling_factor,
                            float theta);

std::tuple<torch::Tensor, torch::Tensor> apply_rotary_pos_emb(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& cos_sin,
    bool interleaved);
}  // namespace detail

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
                  int64_t max_position_embeddings,
                  float scaling_factor,
                  float rope_theta,
                  bool interleaved,
                  const torch::TensorOptions& options);
};

// ============= Rotary positional embedding implementations =============
// Two types of rotary positional embeddings:
// 1> Interleaved rotation style: rotates pairs of even and odd dimensions
// (used in GPT-J and LLama).
// 2> Half-half rotation style of rotary positional embedding (as seen in
// GPT-Neo).
class RotaryEmbeddingGeneric : public RotaryEmbeddingImpl {
 public:
  RotaryEmbeddingGeneric(int64_t rotary_dim,
                         int64_t max_position_embeddings,
                         float scaling_factor,
                         float theta,
                         bool interleaved,
                         const torch::TensorOptions& options);

  // inplace rotary positional embedding
  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& query,     // [num_tokens, n_heads, head_dim]
      const torch::Tensor& key,       // [num_tokens, n_kv_heads, head_dim]
      const torch::Tensor& positions  // [num_tokens]
  ) const override;

 private:
  torch::Tensor cos_sin_cache_;

  int64_t rotary_dim_ = 0;

  bool interleaved_ = false;
};

class RotaryEmbeddingKernel : public RotaryEmbeddingImpl {
 public:
  RotaryEmbeddingKernel(int64_t rotary_dim,
                        int64_t max_position_embeddings,
                        float scaling_factor,
                        float theta,
                        bool interleaved,
                        const torch::TensorOptions& options);

  // inplace rotary positional embedding
  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& query,     // [num_tokens, n_heads, head_dim]
      const torch::Tensor& key,       // [num_tokens, n_kv_heads, head_dim]
      const torch::Tensor& positions  // [num_tokens]
  ) const override;

 private:
  torch::Tensor cos_sin_cache_;

  int64_t rotary_dim_ = 0;

  bool interleaved_ = false;
};

}  // namespace llm
