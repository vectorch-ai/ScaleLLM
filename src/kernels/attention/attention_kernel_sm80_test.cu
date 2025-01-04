#include <ATen/cuda/Exceptions.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include "attention_kernel_sm80.cuh"
#include "attention_traits_sm80.h"

namespace llm {
namespace {
// Multi-head attention implementation using pytorch
torch::Tensor attention_ref(
    torch::Tensor query,  // [batch_size, n_heads, q_len, head_dim]
    torch::Tensor key,    // [batch_size, n_kv_heads, kv_len, head_dim]
    torch::Tensor value   // [batch_size, n_kv_heads, kv_len, head_dim]
) {
  const auto n_heads = query.size(1);
  const auto n_kv_heads = key.size(1);
  const auto head_dim = query.size(3);
  assert(n_heads == n_kv_heads);

  const float sm_scale = 1.0 / sqrt(head_dim);
  // query * key => [n_heads, q_seq_len, seq_len]
  auto scores = torch::einsum("bhqd,bhkd->bhqk", {query, key});
  // apply scale
  scores *= sm_scale;

  // safe softmax
  scores = torch::softmax(scores, /*dim=*/-1);

  // score * value => [batch_size, n_heads, q_seq_len, head_dim]
  return torch::einsum("bhqk,bhkd->bhqd", {scores, value});
}

torch::Tensor attention_sm80(
    torch::Tensor query,  // [batch_size, n_heads, q_len, head_dim]
    torch::Tensor key,    // [batch_size, n_kv_heads, kv_len, head_dim]
    torch::Tensor value   // [batch_size, n_kv_heads, kv_len, head_dim]
) {
  const auto batch_size = query.size(0);
  const auto q_len = query.size(2);
  const auto kv_len = key.size(2);
  const auto head_dim = query.size(3);

  const auto h_stride = query.stride(1);
  const auto kv_h_stride = key.stride(1);

  auto out = torch::empty_like(query);

  constexpr int32_t kHeadDim = 64;
  constexpr int32_t kBlockM = 64;
  constexpr int32_t kBlockN = 64;

  const float sm_scale = 1.0 / sqrt(head_dim) * M_LOG2E;

  using AttentionTraits =
      AttentionTraitsSM80<cute::half_t, kHeadDim, kBlockM, kBlockN>;

  dim3 block = AttentionTraits::kThreadNum;
  dim3 grid((q_len + kBlockM - 1) / kBlockM, batch_size * head_dim);

  const auto smem_size = AttentionTraits::kSmemSize;
  auto attention_kernel = mha_kernel_sm80<AttentionTraits>;
  C10_CUDA_CHECK(
      cudaFuncSetAttribute(attention_kernel,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           smem_size));
  attention_kernel<<<grid, block, smem_size>>>(out.mutable_data_ptr(),
                                               query.const_data_ptr(),
                                               key.const_data_ptr(),
                                               value.const_data_ptr(),
                                               h_stride,
                                               kv_h_stride,
                                               q_len,
                                               kv_len,
                                               sm_scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace

class AttentionKernelTest
    : public ::testing::TestWithParam<std::tuple<int64_t /*batch_size*/,
                                                 int64_t /*q_len*/,
                                                 int64_t /*kv_len*/,
                                                 int64_t /*n_heads*/,
                                                 int64_t /*n_kv_heads*/,
                                                 int64_t /*head_dim*/>> {};

TEST_P(AttentionKernelTest, MHA) {
  const auto [batch_size, q_len, kv_len, n_heads, n_kv_heads, head_dim] =
      GetParam();

  const auto options = torch::dtype(torch::kHalf).device(torch::kCUDA);

  const auto query =
      torch::randn({batch_size, n_heads, q_len, head_dim}, options);
  const auto key =
      torch::randn({batch_size, n_kv_heads, kv_len, head_dim}, options);
  const auto value =
      torch::randn({batch_size, n_kv_heads, kv_len, head_dim}, options);

  auto ref_out = attention_ref(query, key, value);
  auto out = attention_sm80(query, key, value);

  EXPECT_TRUE(torch::allclose(out, ref_out, /*rtol=*/1e-3, /*atol=*/1e-3));
}

INSTANTIATE_TEST_SUITE_P(MHA,
                         AttentionKernelTest,
                         ::testing::Combine(::testing::Values(1),  // batch_size
                                            ::testing::Values(64),  // q_len
                                            ::testing::Values(64,
                                                              256),  // kv_len
                                            ::testing::Values(2),    // n_heads
                                            ::testing::Values(2),  // n_kv_heads
                                            ::testing::Values(64)  // head_dim
                                            ));

}  // namespace llm