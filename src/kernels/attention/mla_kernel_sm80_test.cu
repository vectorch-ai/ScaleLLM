#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <cute/layout.hpp>

#include "mla_kernel_sm80.cuh"  // IWYU pragma: keep
#include "mla_params.h"
#include "mla_ref.h"

namespace llm {

namespace {
torch::Tensor mla_sm80(
    torch::Tensor q,       // [batch, q_len, n_heads, kv_lora_rank]
    torch::Tensor q_rope,  // [batch, q_len, n_heads, qk_rope_head_dim]
    torch::Tensor kv,      // [batch, kv_len, kv_lora_rank]
    torch::Tensor k_rope,  // [batch, kv_len, qk_rope_head_dim]
    float sm_scale) {
  // const auto batch_size = query.size(0);
  // const auto q_len = query.size(-3);
  // const auto kv_len = key.size(-3);
  // const auto n_heads = query.size(-2);
  // const auto n_kv_heads = key.size(-2);
  // const auto head_dim = query.size(-1);

  auto out = torch::empty_like(q);

  // const float sm_scale = 1.0 / sqrt(head_dim);

  // // construct attention params
  // MLAParams params;
  // params.q_ptr = query.const_data_ptr();
  // params.q_stride =
  //     make_stride(query.stride(0), query.stride(1), query.stride(2));
  // params.k_ptr = key.const_data_ptr();
  // params.kv_stride = make_stride(key.stride(0), key.stride(1),
  // key.stride(2));

  // params.o_ptr = out.mutable_data_ptr();
  // params.o_stride = make_stride(out.stride(0), out.stride(1), out.stride(2));

  // params.batch_size = batch_size;
  // params.max_q_len = max_q_len;
  // params.n_heads = n_heads;
  // params.q_len = q_len;
  // params.kv_len = kv_len;
  // // params.head_dim = head_dim;

  // // DISPATCH_TORCH_DTYPE_(query.dtype(), DTYPE, [&] {
  // //   DISPATCH_HEAD_DIM_(head_dim, HEAD_DIM, [&] {
  // //     run_mha_kernel_sm80<DTYPE, HEAD_DIM>(params);
  // //   });
  // // });
  return out;
}

}  // namespace

class MLAKernelTest : public ::testing::TestWithParam<
                          std::tuple<torch::ScalarType /*q_dtype*/,
                                     int64_t /*batch_size*/,
                                     int64_t /*q_len*/,
                                     int64_t /*kv_len*/,
                                     int64_t /*n_heads*/,
                                     int64_t /*kv_lora_rank*/,
                                     int64_t /*qk_rope_head_dim*/>> {
 public:
  void SetUp() override {
    // Set random seed for test stability
    torch::manual_seed(0);
  }
};

TEST_P(MLAKernelTest, MLA) {
  const auto [dtype,
              batch_size,
              q_len,
              kv_len,
              n_heads,
              kv_lora_rank,
              qk_rope_head_dim] = GetParam();
  const auto head_dim = kv_lora_rank + qk_rope_head_dim;
  const auto options = torch::dtype(dtype).device(torch::kCUDA);

  const auto q =
      torch::randn({batch_size, q_len, n_heads, kv_lora_rank}, options);
  const auto q_rope =
      torch::randn({batch_size, q_len, n_heads, qk_rope_head_dim}, options);

  const auto kv = torch::randn({batch_size, kv_len, kv_lora_rank}, options);
  const auto k_rope =
      torch::randn({batch_size, kv_len, qk_rope_head_dim}, options);

  const float sm_scale = 1.0 / sqrt(head_dim);

  auto ref_out = mla_batch_ref(q, q_rope, kv, k_rope, sm_scale);
  auto out = mla_sm80(q, q_rope, kv, k_rope, sm_scale);

  if (dtype == torch::kBFloat16) {
    EXPECT_TRUE(torch::allclose(out, ref_out, /*rtol=*/1e-2, /*atol=*/1e-2));
  } else {
    EXPECT_TRUE(torch::allclose(out, ref_out, /*rtol=*/1e-3, /*atol=*/1e-3));
  }
}

INSTANTIATE_TEST_SUITE_P(
    MLA,
    MLAKernelTest,
    ::testing::Combine(::testing::Values(torch::kHalf),  // q_dtype
                       ::testing::Values(1),             // batch_size
                       ::testing::Values(64),            // q_len
                       ::testing::Values(64),            // kv_len
                       ::testing::Values(8),             // n_heads
                       ::testing::Values(64),            // kv_lora_rank
                       ::testing::Values(64)             // qk_rope_head_dim
                       ));

}  // namespace llm