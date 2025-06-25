#include <absl/random/random.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include "cute/layout.hpp"
#include "mla_params.h"
#include "mla_ref.h"
#include "sm80_mla_dispatch.cuh"

namespace llm {
using namespace cute;

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
torch::Tensor mla_pagedkv_sm80(
    torch::Tensor q,              // [q_seq_len, n_heads, head_dim]
    torch::Tensor kv_cache,       // [n_slots, head_dim]
    torch::Tensor q_rope,         // [q_seq_len, n_heads, rope_head_dim]
    torch::Tensor k_rope_cache,   // [n_slots, rope_head_dim]
    torch::Tensor q_cu_lens,      // [batch_size+1]
    torch::Tensor kv_cu_lens,     // [batch_size+1]
    torch::Tensor block_table,    // [n_blocks]
    torch::Tensor block_cu_lens,  // [batch_size+1]
    int block_size,
    int max_q_len,
    float sm_scale) {
  const auto batch_size = q_cu_lens.size(0) - 1;
  const auto n_heads = q.size(-2);
  const auto head_dim = q.size(-1);
  const auto rope_head_dim = q_rope.size(-1);

  auto out = torch::empty_like(q);

  // construct attention params
  MLAPagedKVParams params;
  params.q_ptr = q.const_data_ptr();
  params.q_stride = make_stride(q.stride(0), q.stride(1), _1{});
  params.kv_ptr = kv_cache.const_data_ptr();
  params.kv_stride = make_stride(kv_cache.stride(0), _1{});
  params.q_rope_ptr = q_rope.const_data_ptr();
  params.q_rope_stride = make_stride(q_rope.stride(0), q_rope.stride(1), _1{});
  params.k_rope_ptr = k_rope_cache.const_data_ptr();
  params.k_rope_stride = make_stride(k_rope_cache.stride(0), _1{});

  params.o_ptr = out.mutable_data_ptr();
  params.o_stride = make_stride(out.stride(0), out.stride(1), _1{});

  params.batch_size = batch_size;
  params.max_q_len = max_q_len;
  params.n_heads = n_heads;
  params.head_dim = head_dim;
  params.rope_head_dim = rope_head_dim;
  params.sm_scale = sm_scale;

  params.q_cu_lens = q_cu_lens.const_data_ptr<int32_t>();
  params.kv_cu_lens = kv_cu_lens.const_data_ptr<int32_t>();

  params.block_table = block_table.const_data_ptr<int32_t>();
  params.block_cu_lens = block_cu_lens.const_data_ptr<int32_t>();
  params.block_size = block_size;

  DISPATCH_TORCH_DTYPE_(q.dtype(), DTYPE, [&] { sm80_run_mla<DTYPE>(params); });
  return out;
}

}  // namespace

class MLAKernelPagedKVTest
    : public ::testing::TestWithParam<std::tuple<torch::ScalarType /*q_dtype*/,
                                                 int64_t /*batch_size*/,
                                                 int64_t /*block_size*/,
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

TEST_P(MLAKernelPagedKVTest, PageKV) {
  const auto [dtype,
              batch_size,
              block_size,
              max_q_len,
              max_kv_len,
              n_heads,
              head_dim,
              rope_head_dim] = GetParam();

  const auto options = torch::dtype(dtype).device(torch::kCUDA);

  std::vector<int32_t> block_table_vec;
  std::vector<int32_t> block_cu_lens_vec = {0};
  std::vector<int> slot_ids;

  const int32_t total_blocks = (max_kv_len * batch_size) / block_size + 2;
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

    // assign blocks for each sequence
    const int32_t n_blocks = (kv_len + block_size - 1) / block_size;
    std::vector<int32_t> block_ids;
    block_ids.reserve(n_blocks);
    for (int j = 0; j < n_blocks; ++j) {
      // random assign block size
      const int32_t id =
          absl::Uniform<int>(absl::IntervalClosedOpen, gen, 1, total_blocks);
      // put first slot id of each block into block_table
      block_ids.push_back(id * block_size);
    }
    block_table_vec.insert(
        block_table_vec.end(), block_ids.begin(), block_ids.end());
    block_cu_lens_vec.push_back(block_table_vec.size());

    for (int j = 0; j < kv_len; ++j) {
      const int32_t slot_base = block_ids[j / block_size];
      const int32_t block_offset = j % block_size;
      slot_ids.push_back(slot_base + block_offset);
    }
  }

  // generate q, kv, q_rope, k_rope
  torch::Tensor q = torch::rand({n_q_tokens, n_heads, head_dim}, options);
  const auto n_slots = total_blocks * block_size;
  torch::Tensor kv_cache = torch::rand({n_slots, head_dim}, options);

  torch::Tensor q_rope =
      torch::rand({n_q_tokens, n_heads, rope_head_dim}, options);
  torch::Tensor k_rope_cache = torch::rand({n_slots, rope_head_dim}, options);

  torch::Tensor q_cu_lens = torch::tensor(
      q_cu_lens_vec, torch::dtype(torch::kInt32).device(torch::kCUDA));
  torch::Tensor kv_cu_lens = torch::tensor(
      kv_cu_lens_vec, torch::dtype(torch::kInt32).device(torch::kCUDA));

  torch::Tensor block_table = torch::tensor(
      block_table_vec, torch::dtype(torch::kInt32).device(torch::kCUDA));
  torch::Tensor block_cu_lens = torch::tensor(
      block_cu_lens_vec, torch::dtype(torch::kInt32).device(torch::kCUDA));

  // get combined key and value
  std::vector<torch::Tensor> kvs;
  kvs.reserve(slot_ids.size());
  std::vector<torch::Tensor> k_ropes;
  k_ropes.reserve(slot_ids.size());
  for (int slot_id : slot_ids) {
    // kv = kv_cache[slot_idx, :, :]
    kvs.push_back(kv_cache[slot_id]);
    k_ropes.push_back(k_rope_cache[slot_id]);
  }
  torch::Tensor kv = torch::stack(kvs, /*dim=*/0);
  torch::Tensor k_rope = torch::stack(k_ropes, /*dim=*/0);

  const float sm_scale = 1.0 / sqrt(head_dim + rope_head_dim);

  auto ref_out =
      mla_varlen_ref(q, kv, q_rope, k_rope, q_cu_lens, kv_cu_lens, sm_scale);
  auto out = mla_pagedkv_sm80(q,
                              kv_cache,
                              q_rope,
                              k_rope_cache,
                              q_cu_lens,
                              kv_cu_lens,
                              block_table,
                              block_cu_lens,
                              block_size,
                              max_q_len,
                              sm_scale);

  // std::cerr << "max diff: " << (ref_out - out).abs().max() << std::endl;
  if (dtype == torch::kBFloat16) {
    EXPECT_TRUE(torch::allclose(out, ref_out, /*rtol=*/1e-2, /*atol=*/1e-2));
  } else {
    EXPECT_TRUE(torch::allclose(out, ref_out, /*rtol=*/1e-3, /*atol=*/1e-3));
  }
}

INSTANTIATE_TEST_SUITE_P(
    MLA,
    MLAKernelPagedKVTest,
    ::testing::Combine(::testing::Values(torch::kHalf,
                                         torch::kBFloat16),  // q_dtype
                       ::testing::Values(1, 2, 4),           // batch_size
                       ::testing::Values(1, 8, 64),          // block_size
                       ::testing::Values(1, 125),            // max_q_len
                       ::testing::Values(127, 1000),         // max_kv_len
                       ::testing::Values(1, 8, 128),         // n_heads
                       ::testing::Values(128, 256, 512),     // head_dim
                       ::testing::Values(64)                 // rope_head_dim
                       ));

}  // namespace llm
