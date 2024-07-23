#include "pos_embedding.h"

#include <ATen/ops/allclose.h>
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

TEST(RopeScalingTest, Llama3) {
  const int64_t rotary_dim = 128;
  const float theta = 500000.0f;
  const auto inv_freq = detail::compute_default_inv_freq(rotary_dim, theta);
  auto expected_inv_freq = torch::tensor(
      {1.0000e+00, 8.1462e-01, 6.6360e-01, 5.4058e-01, 4.4037e-01, 3.5873e-01,
       2.9223e-01, 2.3805e-01, 1.9392e-01, 1.5797e-01, 1.2869e-01, 1.0483e-01,
       8.5397e-02, 6.9566e-02, 5.6670e-02, 4.6164e-02, 3.7606e-02, 3.0635e-02,
       2.4955e-02, 2.0329e-02, 1.6560e-02, 1.3490e-02, 1.0990e-02, 8.9523e-03,
       7.2927e-03, 5.9407e-03, 4.8394e-03, 3.9423e-03, 3.2114e-03, 2.6161e-03,
       2.1311e-03, 1.7360e-03, 1.4142e-03, 1.1520e-03, 9.3847e-04, 7.6450e-04,
       6.2277e-04, 5.0732e-04, 4.1327e-04, 3.3666e-04, 2.7425e-04, 2.2341e-04,
       1.8199e-04, 1.4825e-04, 1.2077e-04, 9.8381e-05, 8.0143e-05, 6.5286e-05,
       5.3183e-05, 4.3324e-05, 3.5292e-05, 2.8750e-05, 2.3420e-05, 1.9078e-05,
       1.5542e-05, 1.2660e-05, 1.0313e-05, 8.4015e-06, 6.8440e-06, 5.5752e-06,
       4.5417e-06, 3.6997e-06, 3.0139e-06, 2.4551e-06},
      torch::kFloat32);

  EXPECT_TRUE(torch::allclose(inv_freq, expected_inv_freq, /*rtol=*/1e-04));

  const float factor = 8.0f;
  const float low_freq_factor = 1.0f;
  const float high_freq_factor = 4.0f;
  const int64_t old_context_len = 8192;
  const auto scaled_inv_freq = detail::apply_llama3_rope_scaling(
      inv_freq, factor, low_freq_factor, high_freq_factor, old_context_len);

  auto expected_scaled_inv_freq = torch::tensor(
      {1.0000e+00, 8.1462e-01, 6.6360e-01, 5.4058e-01, 4.4037e-01, 3.5873e-01,
       2.9223e-01, 2.3805e-01, 1.9392e-01, 1.5797e-01, 1.2869e-01, 1.0483e-01,
       8.5397e-02, 6.9566e-02, 5.6670e-02, 4.6164e-02, 3.7606e-02, 3.0635e-02,
       2.4955e-02, 2.0329e-02, 1.6560e-02, 1.3490e-02, 1.0990e-02, 8.9523e-03,
       7.2927e-03, 5.9407e-03, 4.8394e-03, 3.9423e-03, 3.2114e-03, 2.1666e-03,
       1.3719e-03, 8.5675e-04, 5.2485e-04, 3.1269e-04, 1.7851e-04, 9.5562e-05,
       7.7847e-05, 6.3415e-05, 5.1659e-05, 4.2082e-05, 3.4281e-05, 2.7926e-05,
       2.2749e-05, 1.8532e-05, 1.5096e-05, 1.2298e-05, 1.0018e-05, 8.1607e-06,
       6.6479e-06, 5.4155e-06, 4.4115e-06, 3.5937e-06, 2.9275e-06, 2.3848e-06,
       1.9427e-06, 1.5826e-06, 1.2892e-06, 1.0502e-06, 8.5550e-07, 6.9690e-07,
       5.6771e-07, 4.6247e-07, 3.7673e-07, 3.0689e-07},
      torch::kFloat32);
  EXPECT_TRUE(torch::allclose(scaled_inv_freq,
                              expected_scaled_inv_freq,
                              /*rtol=*/1e-04));
}

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

  const auto inv_freq = detail::compute_default_inv_freq(head_dim, theta);
  RotaryEmbeddingGeneric rotary_embedding(
      head_dim, max_position_embeddings, inv_freq, interleaved, options);
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

  const auto inv_freq = detail::compute_default_inv_freq(rotary_dim, theta);
  RotaryEmbeddingGeneric rotary_embedding(
      rotary_dim, max_position_embeddings, inv_freq, interleaved, options);

  RotaryEmbeddingKernel rotary_embedding_kernel(
      rotary_dim, max_position_embeddings, inv_freq, interleaved, options);

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
        ::testing::Values(100000.0f, 500000.0f),              // theta
        ::testing::Values(false, true),                       // interleaved
        ::testing::Values(4096, 8192)  // max_position_embeddings
        ));

}  // namespace llm
