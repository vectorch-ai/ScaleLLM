#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <cute/layout.hpp>

#include "device/sm80_mla_dispatch.cuh"
#include "mla_params.h"
#include "tests/mla_ref.h"

namespace llm {
#define DISPATCH_TORCH_DTYPE_(TORCH_DTYPE, TYPE_NAME, ...) \
  [&] {                                                    \
    if (TORCH_DTYPE == torch::kHalf) {                     \
      using TYPE_NAME = cute::half_t;                      \
      return __VA_ARGS__();                                \
    } else if (TORCH_DTYPE == torch::kBFloat16) {          \
      using TYPE_NAME = cute::bfloat16_t;                  \
      return __VA_ARGS__();                                \
    } else {                                               \
      assert(false);                                       \
    }                                                      \
  }()

namespace {
torch::Tensor mla_sm80(
    torch::Tensor q,       // [batch, q_len, n_heads, head_dim]
    torch::Tensor kv,      // [batch, kv_len, head_dim]
    torch::Tensor q_rope,  // [batch, q_len, n_heads, rope_head_dim]
    torch::Tensor k_rope,  // [batch, kv_len, rope_head_dim]
    float sm_scale) {
  const auto batch_size = q.size(0);
  const auto q_len = q.size(-3);
  const auto kv_len = kv.size(-2);
  const auto n_heads = q.size(-2);
  const auto head_dim = q.size(-1);
  const auto rope_head_dim = q_rope.size(-1);

  auto out = torch::empty_like(q);

  // construct attention params
  MLAParams params;
  params.q_ptr = q.const_data_ptr();
  params.q_stride = make_stride(q.stride(0), q.stride(1), q.stride(2), _1{});
  params.kv_ptr = kv.const_data_ptr();
  params.kv_stride = make_stride(kv.stride(0), kv.stride(1), _1{});

  params.q_rope_ptr = q_rope.const_data_ptr();
  params.q_rope_stride =
      make_stride(q_rope.stride(0), q_rope.stride(1), q_rope.stride(2), _1{});
  params.k_rope_ptr = k_rope.const_data_ptr();
  params.k_rope_stride = make_stride(k_rope.stride(0), k_rope.stride(1), _1{});

  params.o_ptr = out.mutable_data_ptr();
  params.o_stride =
      make_stride(out.stride(0), out.stride(1), out.stride(2), _1{});

  params.batch_size = batch_size;
  params.max_q_len = q_len;
  params.n_heads = n_heads;
  params.q_len = q_len;
  params.kv_len = kv_len;
  params.head_dim = head_dim;
  params.rope_head_dim = rope_head_dim;
  params.sm_scale = sm_scale;
  params.normalize();

  DISPATCH_TORCH_DTYPE_(q.dtype(), DTYPE, [&] { sm80_run_mla<DTYPE>(params); });
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
  // skip invalid test cases
  if (kv_len < q_len) {
    return;
  }

  const auto options = torch::dtype(dtype).device(torch::kCUDA);

  // q: [batch, q_len, n_heads, head_dim]
  // kv: [batch, kv_len, head_dim]
  const auto q = torch::randn({batch_size, q_len, n_heads, head_dim}, options);
  const auto kv = torch::randn({batch_size, kv_len, head_dim}, options);

  // q_rope: [batch, q_len, n_heads, rope_head_dim]
  // kv_rope: [batch, kv_len, rope_head_dim]
  const auto q_rope =
      torch::randn({batch_size, q_len, n_heads, rope_head_dim}, options);
  const auto k_rope =
      torch::randn({batch_size, kv_len, rope_head_dim}, options);

  const float sm_scale = 1.0 / sqrt(head_dim + rope_head_dim);

  auto ref_out = mla_batch_ref(q, kv, q_rope, k_rope, sm_scale);
  auto out = mla_sm80(q, kv, q_rope, k_rope, sm_scale);
  // std::cerr << "max diff: " << (ref_out - out).abs().max() << std::endl;
  if (dtype == torch::kBFloat16) {
    EXPECT_TRUE(torch::allclose(out, ref_out, /*rtol=*/1e-2, /*atol=*/1e-2));
  } else {
    EXPECT_TRUE(torch::allclose(out, ref_out, /*rtol=*/1e-3, /*atol=*/1e-3));
  }
}

INSTANTIATE_TEST_SUITE_P(
    SM80,
    MLAKernelTest,
    ::testing::Combine(::testing::Values(torch::kHalf,
                                         torch::kBFloat16),  // q_dtype
                       ::testing::Values(1, 2, 4, 10),       // batch_size
                       ::testing::Values(1, 62, 125),        // q_len
                       ::testing::Values(1, 30, 287, 1000),  // kv_len
                       ::testing::Values(1, 8, 128),         // n_heads
                       ::testing::Values(128, 256, 512),     // head_dim
                       ::testing::Values(64)                 // rope_head_dim
                       ));

}  // namespace llm
