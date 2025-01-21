#include <absl/random/random.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include "attention_launch_sm80.cuh"
#include "attention_params.h"
#include "cute/layout.hpp"
#include "static_dispatch.h"

namespace llm {
namespace {
// Multi-head attention implementation using pytorch
torch::Tensor attention_ref(
    torch::Tensor query,  // [q_len, n_heads, head_dim]
    torch::Tensor key,    // [kv_len, n_kv_heads, head_dim]
    torch::Tensor value,  // [kv_len, n_kv_heads, head_dim]
    torch::optional<torch::Tensor> alibi_slopes,  //[n_heads]
    float logits_soft_cap,
    int32_t sliding_window) {
  const auto q_len = query.size(-3);
  const auto kv_len = key.size(-3);
  const auto n_heads = query.size(-2);
  const auto n_kv_heads = key.size(-2);
  const auto head_dim = query.size(-1);
  assert(kv_len >= q_len);

  if (n_heads != n_kv_heads) {
    assert(n_heads % n_kv_heads == 0);
    const auto group_size = n_heads / n_kv_heads;
    key = key.repeat_interleave(/*repeats=*/group_size, /*dim=*/-2);
    value = value.repeat_interleave(/*repeats=*/group_size, /*dim=*/-2);
  }

  const float sm_scale = 1.0 / sqrt(head_dim);
  // query * key => [n_heads, q_len, kv_len]
  auto scores = torch::einsum("qhd,khd->hqk",
                              {query.to(torch::kFloat), key.to(torch::kFloat)});
  // apply scale
  scores *= sm_scale;

  // apply softcap if needed
  if (logits_soft_cap != 0.0) {
    scores = torch::tanh(scores / logits_soft_cap) * logits_soft_cap;
  }

  // apply alibi bias
  if (alibi_slopes) {
    const auto& slopes = alibi_slopes.value();
    // calculate alibi attention bias
    // since it's causal mask, we can just use [0, 1, ...,, kv_len)
    auto distance = torch::arange(0, kv_len, query.options());
    // [n_heads, 1, kv_len]
    auto bias = distance.view({1, 1, kv_len}) * slopes.view({n_heads, 1, 1});
    scores += bias;
  }

  auto mask = torch::ones({q_len, kv_len}, torch::kBool);
  if (sliding_window >= 0) {
    // sliding window mask
    // returns the upper triangular part of a matrix
    mask = torch::triu(mask, /*diagonal=*/kv_len - q_len - sliding_window);
  }

  // apply causal mask
  // causal mask: returns the lower triangular part of a matrix
  mask = torch::tril(mask, /*diagonal=*/kv_len - q_len).to(query);
  scores = scores.masked_fill(mask == 0, -INFINITY);

  // safe softmax
  scores = torch::softmax(scores, /*dim=*/-1);

  // score * value => [q_len, n_heads, head_dim]
  return torch::einsum("hqk,khd->qhd", {scores, value.to(torch::kFloat)})
      .type_as(query);
}

torch::Tensor attention_varlen_ref(
    torch::Tensor query,       // [q_len, n_heads, head_dim]
    torch::Tensor key,         // [kv_len, n_kv_heads, head_dim]
    torch::Tensor value,       // [kv_len, n_kv_heads, head_dim]
    torch::Tensor q_cu_lens,   // [batch_size + 1]
    torch::Tensor kv_cu_lens,  // [batch_size + 1]
    torch::optional<torch::Tensor> alibi_slopes,  //[n_heads]
    float logits_soft_cap,
    int32_t sliding_window) {
  torch::Tensor q_cu_lens_cpu = q_cu_lens.cpu();
  torch::Tensor kv_cu_seq_lens_cpu = kv_cu_lens.cpu();
  const size_t n_seqs = q_cu_lens_cpu.numel() - 1;
  const int32_t* q_cu_lens_ptr = q_cu_lens_cpu.data_ptr<int32_t>();
  const int32_t* kv_cu_lens_ptr = kv_cu_seq_lens_cpu.data_ptr<int32_t>();

  std::vector<torch::Tensor> out_list;
  // process sequence one by one
  for (int64_t i = 0; i < n_seqs; ++i) {
    // calaculate attention for each sequence
    const int32_t q_start = q_cu_lens_ptr[i];
    const int32_t q_end = q_cu_lens_ptr[i + 1];
    const int32_t kv_start = kv_cu_lens_ptr[i];
    const int32_t kv_end = kv_cu_lens_ptr[i + 1];

    torch::Tensor q = query.slice(/*dim=*/0, /*start=*/q_start, /*end=*/q_end);
    torch::Tensor k = key.slice(/*dim=*/0, /*start=*/kv_start, /*end=*/kv_end);
    torch::Tensor v =
        value.slice(/*dim=*/0, /*start=*/kv_start, /*end=*/kv_end);

    auto output =
        attention_ref(q, k, v, alibi_slopes, logits_soft_cap, sliding_window);
    out_list.push_back(output);
  }
  return torch::cat(out_list, /*dim=*/0);
}

torch::Tensor attention_varlen_sm80(
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
  VarLenAttentionParams params;
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

  DISPATCH_TORCH_DTYPE(query.dtype(), DTYPE, [&] {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, [&] {
      run_attention_kernel_sm80<DTYPE, DTYPE, HEAD_DIM>(params);
    });
  });
  return out;
}

}  // namespace

class AttentionKernelVarlenTest
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

TEST_P(AttentionKernelVarlenTest, VarLen) {
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

  auto ref_out = attention_varlen_ref(query,
                                      key,
                                      value,
                                      q_cu_lens,
                                      kv_cu_lens,
                                      alibi_slopes,
                                      logits_soft_cap,
                                      sliding_window);
  auto out = attention_varlen_sm80(query,
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
    VarLen,
    AttentionKernelVarlenTest,
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