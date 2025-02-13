#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <cute/layout.hpp>
#include <iostream>

#include "cute/numeric/numeric_types.hpp"
#include "mla_kernel_sm80.cuh"  // IWYU pragma: keep
#include "mla_params.h"
#include "mla_ref.h"
#include "mla_traits_sm80.h"

namespace llm {

namespace {
torch::Tensor mla_sm80(
    torch::Tensor q,       // [batch, q_len, n_heads, head_dim]
    torch::Tensor kv,      // [batch, kv_len, head_dim]
    torch::Tensor q_rope,  // [batch, q_len, n_heads, rope_head_dim]
    torch::Tensor k_rope,  // [batch, kv_len, rope_head_dim]
    float sm_scale) {
  const auto batch_size = q.size(0);
  const auto q_len = q.size(-3);
  const auto kv_len = kv.size(-3);
  const auto n_heads = q.size(-2);
  const auto head_dim = q.size(-1);
  const auto rope_head_dim = q_rope.size(-1);

  auto out = torch::empty_like(q);

  // construct attention params
  MLAParams params;
  params.q_ptr = q.const_data_ptr();
  params.q_stride = make_stride(q.stride(0), q.stride(1), q.stride(2));
  params.kv_ptr = kv.const_data_ptr();
  params.kv_stride = make_stride(kv.stride(0), kv.stride(1));

  params.q_rope_ptr = q_rope.const_data_ptr();
  params.q_rope_stride =
      make_stride(q_rope.stride(0), q_rope.stride(1), q_rope.stride(2));
  params.k_rope_ptr = k_rope.const_data_ptr();
  params.k_rope_stride = make_stride(k_rope.stride(0), k_rope.stride(1));

  params.o_ptr = out.mutable_data_ptr();
  params.o_stride = make_stride(out.stride(0), out.stride(1), out.stride(2));

  params.batch_size = batch_size;
  params.max_q_len = q_len;
  params.n_heads = n_heads;
  params.q_len = q_len;
  params.kv_len = kv_len;
  params.head_dim = head_dim;
  params.rope_head_dim = rope_head_dim;
  params.sm_scale = sm_scale;
  params.normalize();

  using Traits = MLATraitsSM80<cute::half_t,
                               /*HEAD_DIM=*/256,
                               /*ROPE_HEAD_DIM=*/64,
                               /*BLK_M=*/64,
                               /*BLK_N=*/64,
                               /*BLK_K=*/64>;
  launch_mla_kernel_sm80<Traits>(params, nullptr);
  return out;
}

}  // namespace

class MLAKernelTest
    : public ::testing::TestWithParam<std::tuple<torch::ScalarType /*q_dtype*/,
                                                 int64_t /*batch_size*/,
                                                 int64_t /*q_len*/,
                                                 int64_t /*kv_len*/,
                                                 int64_t /*n_heads*/,
                                                 int64_t /*head_dim*/,
                                                 int64_t /*rope_head_dim*/>> {
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
              head_dim,
              rope_head_dim] = GetParam();
  // const auto head_dim = kv_lora_rank + rope_head_dim;
  const auto options = torch::dtype(dtype).device(torch::kCUDA);

  // q: [batch, len, n_heads, head_dim]
  // kv: [batch, len, head_dim]
  const auto q = torch::randn({batch_size, q_len, n_heads, head_dim}, options);
  const auto kv = torch::randn({batch_size, kv_len, head_dim}, options);

  // q_rope: [batch, len, n_heads, rope_head_dim]
  // kv_rope: [batch, len, rope_head_dim]
  const auto q_rope =
      torch::randn({batch_size, q_len, n_heads, rope_head_dim}, options);
  const auto k_rope =
      torch::randn({batch_size, kv_len, rope_head_dim}, options);

  const float sm_scale = 1.0 / sqrt(head_dim + rope_head_dim);

  auto ref_out = mla_batch_ref(q, kv, q_rope, k_rope, sm_scale);
  auto out = mla_sm80(q, kv, q_rope, k_rope, sm_scale);
  std::cerr << "max diff: " << (ref_out - out).abs().max() << std::endl;
  // std::cerr << "ref_out: " << ref_out << std::endl;
  // std::cerr << "out: " << out << std::endl;
  EXPECT_TRUE(torch::allclose(out, ref_out, /*rtol=*/1e-3, /*atol=*/1e-3));
}

INSTANTIATE_TEST_SUITE_P(
    MLA,
    MLAKernelTest,
    ::testing::Combine(::testing::Values(torch::kHalf),  // q_dtype
                       ::testing::Values(1),             // batch_size
                       ::testing::Values(64),            // q_len
                       ::testing::Values(64),            // kv_len
                       ::testing::Values(1),             // n_heads
                       ::testing::Values(256),           // head_dim
                       ::testing::Values(64)             // rope_head_dim
                       ));

}  // namespace llm