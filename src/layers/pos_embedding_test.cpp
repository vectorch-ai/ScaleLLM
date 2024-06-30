#include "pos_embedding.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <tuple>

namespace llm {
namespace {
// Rotary code ported from llama repo, which is used as disired output
torch::Tensor precompute_freqs_cis(int64_t dim,
                                   int64_t max_position_embeddings,
                                   float theta) {
  auto range =
      torch::arange(/*start=*/0, /*end=*/dim, /*step=*/2, torch::kFloat32);
  auto slice = range.slice(/*dim=*/0, /*start=*/0, /*end=*/dim / 2);
  auto freqs = 1.0 / torch::pow(theta, slice / dim);
  auto t = torch::arange(/*end=*/max_position_embeddings, torch::kFloat32);
  freqs = torch::outer(t, freqs);
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
  using torch::indexing::None;
  using ISlice = torch::indexing::Slice;
  auto x1 = x.index({"...", ISlice(0, None, 2)});
  auto x2 = x.index({"...", ISlice(1, None, 2)});
  return torch::cat({x1, x2}, /*dim=*/-1);
}

// [1, 3, 5, 2, 4, 6] -> [1, 2, 3, 4, 5, 6]
inline torch::Tensor half_to_interleaved(const torch::Tensor& x) {
  auto chunks = x.chunk(2, /*dim=*/-1);
  return torch::stack({chunks[0], chunks[1]}, /*dim=*/-1)
      .flatten(/*start_dim=*/-2);
}

std::tuple<torch::Tensor, torch::Tensor> apply_rotary_emb_ref(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& positions,
    int64_t head_dim,
    int64_t max_position_embeddings,
    float theta,
    bool interleaved) {
  auto freqs_cis =
      precompute_freqs_cis(head_dim, max_position_embeddings, theta);
  namespace F = torch::nn::functional;
  auto selected_freqs_cis = F::embedding(positions, freqs_cis);

  if (interleaved) {
    return apply_rotary_emb(query, key, selected_freqs_cis);
  }

  auto interleaved_query = half_to_interleaved(query);
  auto interleaved_key = half_to_interleaved(key);
  auto [query_ref, key_ref] =
      apply_rotary_emb(interleaved_query, interleaved_key, selected_freqs_cis);
  query_ref = interleaved_to_half(query_ref);
  key_ref = interleaved_to_half(key_ref);
  return std::make_tuple(query_ref, key_ref);
}

}  // namespace

class PosEmbeddingTest : public ::testing::TestWithParam<
                             std::tuple<torch::Device,
                                        torch::ScalarType,
                                        int64_t /*num_tokens*/,
                                        int64_t /*n_heads*/,
                                        int64_t /*n_kv_heads*/,
                                        int64_t /*head_dim*/,
                                        float /*theta*/,
                                        bool /*interleaved*/,
                                        int64_t /*max_position_embeddings*/>> {
};

TEST_P(PosEmbeddingTest, Rotary) {
  const auto [device,
              dtype,
              num_tokens,
              n_heads,
              n_kv_heads,
              head_dim,
              theta,
              interleaved,
              max_position_embeddings] = GetParam();
  if (device.is_cuda() && !torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA not available, skipping test";
  }

  const auto options = torch::dtype(dtype).device(device);

  // prepare inputs
  torch::Tensor query = torch::rand({num_tokens, n_heads, head_dim}, options);
  torch::Tensor key = torch::rand({num_tokens, n_kv_heads, head_dim}, options);
  const torch::Tensor positions = torch::randint(
      0, max_position_embeddings, {num_tokens}, options.dtype(torch::kInt));

  RotaryEmbeddingGeneric rotary_embedding(head_dim,
                                          max_position_embeddings,
                                          /*scaling_factor*/ 0.0f,
                                          theta,
                                          interleaved,
                                          options);
  const auto [query_output, key_output] =
      rotary_embedding.forward(query, key, positions);

  // compute the desired output
  auto [query_ref, key_ref] = apply_rotary_emb_ref(query,
                                                   key,
                                                   positions,
                                                   head_dim,
                                                   max_position_embeddings,
                                                   theta,
                                                   interleaved);

  ASSERT_TRUE(torch::allclose(query_ref,
                              query_output,
                              /*rtol=*/1e-03,
                              /*atol=*/1e-05));
  ASSERT_TRUE(torch::allclose(key_ref,
                              key_output,
                              /*rtol=*/1e-03,
                              /*atol=*/1e-05));
}

INSTANTIATE_TEST_SUITE_P(
    RotaryCorrectness,
    PosEmbeddingTest,
    ::testing::Combine(
        ::testing::Values(torch::kCPU),
        ::testing::Values(torch::kFloat),
        ::testing::Values(1, 2, 8, 16),                       // num_tokens
        ::testing::Values(32),                                // n_heads
        ::testing::Values(32 /*mha*/, 8 /*gqa*/, 1 /*mqa*/),  // n_kv_heads
        ::testing::Values(128),                               // head_dim
        ::testing::Values(100000.0f, 500000.0f),              // theta
        ::testing::Values(false, true),                       // interleaved
        ::testing::Values(4096, 8192)  // max_position_embeddings
        ));

class PosEmbeddingKernelTest
    : public ::testing::TestWithParam<
          std::tuple<torch::Device,
                     torch::ScalarType,
                     int64_t /*num_tokens*/,
                     int64_t /*n_heads*/,
                     int64_t /*n_kv_heads*/,
                     int64_t /*head_dim*/,
                     int64_t /*rotary_dim*/,
                     float /*scaling_factor*/,
                     float /*theta*/,
                     bool /*interleaved*/,
                     int64_t /*max_position_embeddings*/>> {};

TEST_P(PosEmbeddingKernelTest, Rotary) {
  const auto [device,
              dtype,
              num_tokens,
              n_heads,
              n_kv_heads,
              head_dim,
              rotary_dim,
              scaling_factor,
              theta,
              interleaved,
              max_position_embeddings] = GetParam();

  if (device.is_cuda() && !torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA not available, skipping test";
  }

  const auto options = torch::dtype(dtype).device(device);
  // prepare inputs
  torch::Tensor query = torch::rand({num_tokens, n_heads, head_dim}, options);
  torch::Tensor key = torch::rand({num_tokens, n_kv_heads, head_dim}, options);
  const torch::Tensor positions = torch::randint(
      0, max_position_embeddings, {num_tokens}, options.dtype(torch::kInt));

  RotaryEmbeddingGeneric rotary_embedding(rotary_dim,
                                          max_position_embeddings,
                                          scaling_factor,
                                          10000.0f,
                                          interleaved,
                                          options);

  RotaryEmbeddingKernel rotary_embedding_kernel(rotary_dim,
                                                max_position_embeddings,
                                                scaling_factor,
                                                10000.0f,
                                                interleaved,
                                                options);

  auto [query_output, key_output] =
      rotary_embedding.forward(query, key, positions);

  // apply rotary embedding using the kernel in place
  auto [query_output_kernel, key_output_kernel] =
      rotary_embedding_kernel.forward(query.clone(), key.clone(), positions);

  ASSERT_TRUE(torch::allclose(query_output, query_output_kernel));
  ASSERT_TRUE(torch::allclose(key_output, key_output_kernel));
}

INSTANTIATE_TEST_SUITE_P(
    Rotary,
    PosEmbeddingKernelTest,
    ::testing::Combine(
        ::testing::Values(torch::kCUDA),
        ::testing::Values(torch::kHalf, torch::kBFloat16),
        ::testing::Values(1, 2, 8, 16),                       // num_tokens
        ::testing::Values(32),                                // n_heads
        ::testing::Values(32 /*mha*/, 8 /*gqa*/, 1 /*mqa*/),  // n_kv_heads
        ::testing::Values(128),                               // head_dim
        ::testing::Values(128, 64),                           // rotary_dim
        ::testing::Values(0.0f, 0.5f),                        // scaling_factor
        ::testing::Values(100000.0f, 500000.0f),              // theta
        ::testing::Values(false, true),                       // interleaved
        ::testing::Values(4096, 8192)  // max_position_embeddings
        ));

}  // namespace llm
