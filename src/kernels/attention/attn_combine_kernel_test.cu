#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch/types.h>

#include <cstdint>

#include "attn_combine_kernel.cuh"  // IWYU pragma: keep

namespace llm {

using namespace cute;

#define DISPATCH_HEAD_DIM_(HEAD_DIM_V, HEAD_DIM_NAME, ...) \
  [&] {                                                    \
    if (HEAD_DIM_V <= 64) {                                \
      constexpr static int HEAD_DIM_NAME = 64;             \
      return __VA_ARGS__();                                \
    } else if (HEAD_DIM_V <= 256) {                        \
      constexpr static int HEAD_DIM_NAME = 256;            \
      return __VA_ARGS__();                                \
    } else {                                               \
      assert(false);                                       \
    }                                                      \
  }()

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

std::tuple<torch::Tensor, torch::Tensor> attn_combine_ref(
    torch::Tensor out_accum,  // [n_splits, batch, seq_len, n_heads, head_dim]
    torch::Tensor lse_accum,  // [n_splits, batch, seq_len, n_heads]
    torch::ScalarType dtype) {
  auto scales = torch::softmax(lse_accum, /*dim=*/0);
  auto lse = torch::logsumexp(lse_accum, /*dim=*/0);
  auto out = torch::einsum("nbshd,nbsh->bshd", {out_accum, scales});
  return {out.to(dtype), lse};
}

struct CombineParams {
  const void* __restrict__ lse_accum_ptr = nullptr;
  const void* __restrict__ o_accum_ptr = nullptr;

  void* __restrict__ o_ptr = nullptr;
  void* __restrict__ lse_ptr = nullptr;

  // input shapes
  int n_splits = 0;
  int batch_size = 0;
  int q_len = 0;
  int n_heads = 0;
  int head_dim = 0;

  // strides
  // [n_splits, batch, seq_len, n_heads, head_dim]
  using OAccumStride = cute::Stride<int64_t, int64_t, int64_t, int64_t /*,_1*/>;
  // [n_splits, batch, seq_len, n_heads]
  using LseAccumStride = cute::Stride<int64_t, int64_t, int64_t /*,_1*/>;
  // [batch, seq_len, n_heads, head_dim]
  using OStride = cute::Stride<int64_t, int64_t, int64_t /*,_1*/>;
  // [batch, seq_len, n_heads]
  using LseStride = cute::Stride<int64_t, int64_t /*,_1*/>;

  OAccumStride o_accum_stride;
  LseAccumStride lse_accum_stride;

  OStride o_stride;
  LseStride lse_stride;
};

std::tuple<torch::Tensor, torch::Tensor> attn_combine(
    torch::Tensor out_accum,  // [n_splits, batch, seq_len, n_heads, head_dim]
    torch::Tensor lse_accum,  // [n_splits, batch, seq_len, n_heads]
    torch::ScalarType dtype) {
  // return out_accum.to(dtype);
  const auto n_splits = out_accum.size(0);
  const auto batch_size = out_accum.size(1);
  const auto q_len = out_accum.size(2);
  const auto n_heads = out_accum.size(3);
  const auto head_dim = out_accum.size(4);

  auto out = torch::empty({batch_size, q_len, n_heads, head_dim},
                          out_accum.options().dtype(dtype));
  auto lse = torch::empty({batch_size, q_len, n_heads}, lse_accum.options());

  CombineParams params;
  params.n_splits = n_splits;
  params.batch_size = batch_size;
  params.q_len = q_len;
  params.n_heads = n_heads;
  params.head_dim = head_dim;

  params.lse_accum_ptr = lse_accum.const_data_ptr();
  params.lse_accum_stride = make_stride(
      lse_accum.stride(0), lse_accum.stride(1), lse_accum.stride(2));

  params.o_accum_ptr = out_accum.const_data_ptr();
  params.o_accum_stride = make_stride(out_accum.stride(0),
                                      out_accum.stride(1),
                                      out_accum.stride(2),
                                      out_accum.stride(3));

  params.o_ptr = out.mutable_data_ptr();
  params.o_stride = make_stride(out.stride(0), out.stride(1), out.stride(2));
  params.lse_ptr = lse.mutable_data_ptr();
  params.lse_stride = make_stride(lse.stride(0), lse.stride(1));

  launch_attn_combine_kernel<cute::half_t,
                             float,
                             /*kHeadDim=*/128,
                             /*kSplits=*/32>(params, nullptr);

  return {out, lse};
}

}  // namespace

class AttnCombineKernelTest
    : public ::testing::TestWithParam<std::tuple<torch::ScalarType /*q_dtype*/,
                                                 int64_t /*n_splits*/,
                                                 int64_t /*batch_size*/,
                                                 int64_t /*q_len*/,
                                                 int64_t /*n_heads*/,
                                                 int64_t /*head_dim*/>> {
 public:
  void SetUp() override {
    // Set random seed for test stability
    torch::manual_seed(0);
  }
};

TEST_P(AttnCombineKernelTest, Combine) {
  const auto [dtype, n_splits, batch_size, q_len, n_heads, head_dim] =
      GetParam();

  const auto options = torch::dtype(torch::kFloat32).device(torch::kCUDA);

  const auto out_accum =
      torch::randn({n_splits, batch_size, q_len, n_heads, head_dim}, options);

  const auto lsm_accum =
      torch::randn({n_splits, batch_size, q_len, n_heads}, options);

  auto [ref_out, ref_lse] = attn_combine_ref(out_accum, lsm_accum, dtype);
  auto [out, lse] = attn_combine(out_accum, lsm_accum, dtype);

  EXPECT_TRUE(torch::allclose(out, ref_out, /*rtol=*/1e-3, /*atol=*/1e-3));
  EXPECT_TRUE(torch::allclose(lse, ref_lse, /*rtol=*/1e-3, /*atol=*/1e-3));
}

INSTANTIATE_TEST_SUITE_P(
    Combine,
    AttnCombineKernelTest,
    ::testing::Combine(::testing::Values(torch::kHalf),   // q_dtype
                       ::testing::Values(2, 4, 10),       // n_splits
                       ::testing::Values(1, 2, 4, 16),    // batch_size
                       ::testing::Values(1, 10, 20),      // q_len
                       ::testing::Values(1, 2, 16, 128),  // n_heads
                       ::testing::Values(128)             // head_dim
                       ));

}  // namespace llm