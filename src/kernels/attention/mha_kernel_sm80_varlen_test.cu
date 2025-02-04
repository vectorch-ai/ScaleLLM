#include <absl/random/random.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include "cute/layout.hpp"
#include "mha_kernel_sm80.cuh"
#include "mha_launch_sm80.cuh"
#include "mha_params.h"
#include "mha_ref.h"

namespace llm {
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

namespace {
torch::Tensor mha_varlen_sm80(
    torch::Tensor query,       // [q_len, n_heads, head_dim]
    torch::Tensor key,         // [kv_len, n_kv_heads, head_dim]
    torch::Tensor value,       // [kv_len, n_kv_heads, head_dim]
    torch::Tensor q_cu_lens,   // [batch_size+1]
    torch::Tensor kv_cu_lens,  // [batch_size+1]
    torch::optional<torch::Tensor> alibi_slopes,  //[n_heads]
    float logits_soft_cap,
    int32_t sliding_window,
    int32_t max_q_len) {
  const auto batch_size = q_cu_lens.size(0) - 1;
  const auto n_heads = query.size(-2);
  const auto n_kv_heads = key.size(-2);
  const auto head_dim = query.size(-1);

  auto out = torch::empty_like(query);

  const float sm_scale = 1.0 / sqrt(head_dim);

  // construct attention params
  MHAVarLenParams params;
  params.q_ptr = query.const_data_ptr();
  params.q_stride = make_stride(query.stride(0), query.stride(1));
  params.k_ptr = key.const_data_ptr();
  params.k_stride = make_stride(key.stride(0), key.stride(1));
  params.v_ptr = value.const_data_ptr();
  params.v_stride = make_stride(value.stride(0), value.stride(1));
  params.o_ptr = out.mutable_data_ptr();
  params.o_stride = make_stride(out.stride(0), out.stride(1));
  params.alibi_slopes_ptr = alibi_slopes.has_value()
                                ? alibi_slopes.value().const_data_ptr<float>()
                                : nullptr;
  params.batch_size = batch_size;
  params.max_q_len = max_q_len;
  params.n_heads = n_heads;
  params.n_kv_heads = n_kv_heads;
  params.head_dim = head_dim;
  params.sm_scale = sm_scale;
  params.logits_soft_cap = logits_soft_cap;
  params.sliding_window = sliding_window;

  params.q_cu_lens = q_cu_lens.const_data_ptr<int32_t>();
  params.kv_cu_lens = kv_cu_lens.const_data_ptr<int32_t>();

  DISPATCH_HEAD_DIM_(head_dim, HEAD_DIM, [&] {
    run_mha_kernel_sm80<cute::half_t, HEAD_DIM>(params);
  });
  return out;
}

}  // namespace

class MHAKernelVarlenTest
    : public ::testing::TestWithParam<std::tuple<int64_t /*batch_size*/,
                                                 int64_t /*q_len*/,
                                                 int64_t /*kv_len*/,
                                                 int64_t /*n_heads*/,
                                                 int64_t /*n_kv_heads*/,
                                                 int64_t /*head_dim*/,
                                                 float /*logits_soft_cap*/,
                                                 bool /*alibi*/,
                                                 int32_t /*sliding_window*/>> {
 public:
  void SetUp() override {
    // Set random seed for test stability
    torch::manual_seed(0);
  }
};

TEST_P(MHAKernelVarlenTest, VarLen) {
  const auto [batch_size,
              max_q_len,
              max_kv_len,
              n_heads,
              n_kv_heads,
              head_dim,
              logits_soft_cap,
              alibi,
              sliding_window] = GetParam();

  const auto options = torch::dtype(torch::kHalf).device(torch::kCUDA);

  // random generate seq lens with size in [1, max_seq_len]
  std::vector<int32_t> q_cu_lens_vec = {0};
  std::vector<int32_t> kv_cu_lens_vec = {0};
  int32_t n_kv_tokens = 0;
  int32_t n_q_tokens = 0;
  absl::BitGen gen;
  for (int i = 0; i < batch_size; ++i) {
    // q_len: [1, q_max_seq_len]
    const int32_t q_len =
        absl::Uniform<int>(absl::IntervalClosedClosed, gen, 1, max_q_len);
    n_q_tokens += q_len;
    q_cu_lens_vec.push_back(n_q_tokens);

    // kv_len >= q_len
    int32_t kv_len = q_len;
    if (q_len < max_kv_len) {
      // sample kv_len from [q_len, kv_max_seq_len]
      kv_len = absl::Uniform<int>(
          absl::IntervalClosedClosed, gen, q_len, max_kv_len);
    }
    n_kv_tokens += kv_len;
    kv_cu_lens_vec.push_back(n_kv_tokens);
    assert(kv_len >= q_len);
  }

  // construct non-contiguous query, key and value
  // generate query, key and value
  torch::Tensor query = torch::rand({n_q_tokens, n_heads, head_dim}, options);
  torch::Tensor key = torch::rand({n_kv_tokens, n_kv_heads, head_dim}, options);
  torch::Tensor value =
      torch::rand({n_kv_tokens, n_kv_heads, head_dim}, options);

  torch::Tensor q_cu_lens = torch::tensor(
      q_cu_lens_vec, torch::dtype(torch::kInt32).device(torch::kCUDA));
  torch::Tensor kv_cu_lens = torch::tensor(
      kv_cu_lens_vec, torch::dtype(torch::kInt32).device(torch::kCUDA));

  torch::optional<torch::Tensor> alibi_slopes;
  if (alibi) {
    alibi_slopes = torch::rand(
        {n_heads}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  }

  auto ref_out = mha_varlen_ref(query,
                                key,
                                value,
                                q_cu_lens,
                                kv_cu_lens,
                                alibi_slopes,
                                logits_soft_cap,
                                sliding_window);
  auto out = mha_varlen_sm80(query,
                             key,
                             value,
                             q_cu_lens,
                             kv_cu_lens,
                             alibi_slopes,
                             logits_soft_cap,
                             sliding_window,
                             max_q_len);

  EXPECT_TRUE(torch::allclose(out, ref_out, /*rtol=*/1e-3, /*atol=*/1e-3));
}

INSTANTIATE_TEST_SUITE_P(
    MHA,
    MHAKernelVarlenTest,
    ::testing::Combine(
        ::testing::Values(1, 2, 4),                          // batch_size
        ::testing::Values(1, 62, 125),                       // max_q_len
        ::testing::Values(127, 287, 1000),                   // max_kv_len
        ::testing::Values(6),                                // n_heads
        ::testing::Values(6 /*mha*/, 3 /*gqa*/, 1 /*mqa*/),  // n_kv_heads
        ::testing::Values(32, 64, 96, 128, 256),             // head_dim
        ::testing::Values(0.0, 50.0),                        // logits_soft_cap
        ::testing::Values(false, true),                      // alibi slope
        ::testing::Values(-1, 0, 10)                         // sliding window
        ));
}  // namespace llm