#include "pos_embedding.h"

#include <c10/core/Device.h>
#include <c10/core/TensorImpl.h>
#include <gtest/gtest.h>

namespace llm {
namespace {
using torch::indexing::None;
using torch::indexing::Slice;

// Rotary code ported from llama repo, which is used as disired output
torch::Tensor precompute_freqs_cis(int64_t dim,
                                   int64_t max_seq_len,
                                   float theta = kDefaultTheta) {
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
std::tuple<torch::Tensor, torch::Tensor> apply_rotary_emb(
    const torch::Tensor& xq,
    const torch::Tensor& xk,
    torch::Tensor freqs_cis) {
  // (num_tokens, n_local_heads, head_dim/2, 2)
  //  -> (num_tokens, n_local_heads, head_dim/2)
  auto xq_complex =
      torch::view_as_complex(split_tensor_by_last_dim(xq.to(torch::kFloat32)));
  auto xk_complex =
      torch::view_as_complex(split_tensor_by_last_dim(xk.to(torch::kFloat32)));

  // reshape for broadcast at n_heads dim => (num_tokens, 1 (n_heads),
  // head_dim/2)
  freqs_cis = freqs_cis.unsqueeze(1);
  // -> (num_tokens, n_heads, head_dim)
  auto xq_out = torch::view_as_real(xq_complex * freqs_cis).flatten(2);
  auto xk_out = torch::view_as_real(xk_complex * freqs_cis).flatten(2);
  return std::make_tuple(xq_out.type_as(xq), xk_out.type_as(xk));
}

// [1, 2, 3, 4, 5, 6] => [1, 3, 5, 2, 4, 6]
inline torch::Tensor interleaved_to_half(const torch::Tensor& x) {
  auto x1 = x.index({Slice(), Slice(), Slice(0, None, 2)});
  auto x2 = x.index({Slice(), Slice(), Slice(1, None, 2)});
  return torch::cat({x1, x2}, /*dim=*/-1);
}

// [1, 3, 5, 2, 4, 6] -> [1, 2, 3, 4, 5, 6]
inline torch::Tensor half_to_interleaved(const torch::Tensor& x) {
  auto chunks = x.chunk(2, /*dim=*/-1);
  return torch::stack({chunks[0], chunks[1]}, /*dim=*/-1)
      .flatten(/*start_dim=*/-2);
}

}  // namespace

TEST(RotaryEmbeddingTest, Interleaved) {
  const int64_t num_tokens = 16;
  const int64_t n_heads = 4;
  const int64_t head_dim = 4;
  const int64_t max_seq_len = 128;
  torch::Device device(torch::kCPU);
  InterleavedRotaryEmbedding rotary_embedding(head_dim, max_seq_len, 0.0f, device);

  torch::Tensor query = torch::rand({num_tokens, n_heads, head_dim});
  torch::Tensor key = torch::rand({num_tokens, n_heads, head_dim});
  const torch::Tensor positions = torch::randint(0, max_seq_len, {num_tokens});

  // make a copy for inplace operation
  const auto [query_output, key_output] =
      rotary_embedding.forward(query, key, positions);

  // compute the desired output
  auto freqs_cis = precompute_freqs_cis(head_dim, max_seq_len);
  namespace F = torch::nn::functional;
  auto selected_freqs_cis = F::embedding(positions, freqs_cis);
  const auto [desired_query, desired_key] =
      apply_rotary_emb(query, key, selected_freqs_cis);

  // check the output
  ASSERT_TRUE(torch::allclose(desired_query,
                              query_output,
                              /*rtol=*/1e-03,
                              /*atol=*/1e-05));
  ASSERT_TRUE(torch::allclose(desired_key,
                              key_output,
                              /*rtol=*/1e-03,
                              /*atol=*/1e-05));
}

TEST(RotaryEmbeddingTest, HalfRotated) {
  const int64_t num_tokens = 16;
  const int64_t n_heads = 4;
  const int64_t head_dim = 4;
  const int64_t max_seq_len = 128;
  torch::Device device(torch::kCPU);
  RotatedRotaryEmbedding rotary_embedding(head_dim, max_seq_len, 0.0f, device);

  torch::Tensor query = torch::rand({num_tokens, n_heads, head_dim});
  torch::Tensor key = torch::rand({num_tokens, n_heads, head_dim});
  const torch::Tensor positions = torch::randint(0, max_seq_len, {num_tokens});

  // make a copy for inplace operation
  const auto [query_output, key_output] =
      rotary_embedding.forward(query, key, positions);

  // compute the desired output
  auto freqs_cis = precompute_freqs_cis(head_dim, max_seq_len);
  namespace F = torch::nn::functional;
  auto selected_freqs_cis = F::embedding(positions, freqs_cis);
  auto [desired_query, desired_key] = apply_rotary_emb(
      half_to_interleaved(query), half_to_interleaved(key), selected_freqs_cis);

  desired_query = interleaved_to_half(desired_query);
  desired_key = interleaved_to_half(desired_key);

  // check the output
  ASSERT_TRUE(torch::allclose(desired_query,
                              query_output,
                              /*rtol=*/1e-03,
                              /*atol=*/1e-05));
  ASSERT_TRUE(torch::allclose(desired_key,
                              key_output,
                              /*rtol=*/1e-03,
                              /*atol=*/1e-05));
}

}  // namespace llm
