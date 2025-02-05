#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>

#include "cute/layout.hpp"
#include "mla_kernel_sm80.cuh"  // IWYU pragma: keep
#include "mla_params.h"
#include "mla_ref.h"

namespace llm {

namespace {
torch::Tensor mla_sm80(
    torch::Tensor query,  // [batch_size, q_len, n_heads, head_dim]
    torch::Tensor key,    // [batch_size, kv_len, n_kv_heads, head_dim]
    torch::Tensor value,  // [batch_size, kv_len, n_kv_heads, head_dim]
    torch::optional<torch::Tensor> alibi_slopes,  //[n_heads]
    float logits_soft_cap,
    int32_t sliding_window,
    int32_t max_q_len) {
  const auto batch_size = query.size(0);
  const auto q_len = query.size(-3);
  const auto kv_len = key.size(-3);
  const auto n_heads = query.size(-2);
  const auto n_kv_heads = key.size(-2);
  const auto head_dim = query.size(-1);

  auto out = torch::empty_like(query);

  const float sm_scale = 1.0 / sqrt(head_dim);

  // construct attention params
  MLAParams params;
  params.q_ptr = query.const_data_ptr();
  params.q_stride =
      make_stride(query.stride(0), query.stride(1), query.stride(2));
  params.k_ptr = key.const_data_ptr();
  params.kv_stride = make_stride(key.stride(0), key.stride(1), key.stride(2));

  params.o_ptr = out.mutable_data_ptr();
  params.o_stride = make_stride(out.stride(0), out.stride(1), out.stride(2));

  params.batch_size = batch_size;
  params.max_q_len = max_q_len;
  params.n_heads = n_heads;
  params.q_len = q_len;
  params.kv_len = kv_len;
  // params.head_dim = head_dim;

  // DISPATCH_TORCH_DTYPE_(query.dtype(), DTYPE, [&] {
  //   DISPATCH_HEAD_DIM_(head_dim, HEAD_DIM, [&] {
  //     run_mha_kernel_sm80<DTYPE, HEAD_DIM>(params);
  //   });
  // });
  return out;
}

}  // namespace

class MLAKernelTest
    : public ::testing::TestWithParam<std::tuple<torch::ScalarType /*q_dtype*/,
                                                 int64_t /*batch_size*/,
                                                 int64_t /*q_len*/,
                                                 int64_t /*kv_len*/,
                                                 int64_t /*n_heads*/,
                                                 int64_t /*head_dim*/>> {
 public:
  void SetUp() override {
    // Set random seed for test stability
    torch::manual_seed(0);
  }
};

TEST_P(MLAKernelTest, MLA) {
  const auto [dtype, batch_size, q_len, kv_len, n_heads, head_dim] = GetParam();

  const auto options = torch::dtype(dtype).device(torch::kCUDA);

  // construct non-contiguous query, key and value
  // const auto data = torch::randn(
  //     {batch_size, q_len, n_heads + 2 * n_kv_heads, head_dim}, options);
  // const auto qkv =
  //     data.split(/*split_size=*/{n_heads, n_kv_heads, n_kv_heads}, /*dim=*/2);
  // const auto& query = qkv[0];
  // const auto& key = qkv[1];
  // const auto& value = qkv[2];

  // auto ref_out = mla_batch_ref(
  //     query, key, value, alibi_slopes, logits_soft_cap, sliding_window);
  // auto out = mla_sm80(
  //     query, key, value, alibi_slopes, logits_soft_cap, sliding_window, q_len);

  // if (dtype == torch::kBFloat16) {
  //   EXPECT_TRUE(torch::allclose(out, ref_out, /*rtol=*/1e-2, /*atol=*/1e-2));
  // } else {
  //   EXPECT_TRUE(torch::allclose(out, ref_out, /*rtol=*/1e-3, /*atol=*/1e-3));
  // }
}

INSTANTIATE_TEST_SUITE_P(
    MLA,
    MLAKernelTest,
    ::testing::Combine(::testing::Values(torch::kHalf),  // q_dtype
                       ::testing::Values(1),             // batch_size
                       ::testing::Values(64),            // q_len
                       ::testing::Values(64),            // kv_len
                       ::testing::Values(8),             // n_heads
                       ::testing::Values(64)             // head_dim
                       ));

}  // namespace llm