
#include <ATen/ops/equal.h>
#include <absl/random/random.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/Optional.h>
#include <glog/logging.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/cuda.h>
#include <torch/torch.h>
#include <torch/types.h>

#include <cstdint>

#include "gtest/gtest.h"
#include "models/parameters.h"
#include "ref_handler.h"
#include "scale_attn_handler.h"

namespace llm {

// Test attention with kv-cache for decode stage
class AttentionDecodeTest
    : public ::testing::TestWithParam<std::tuple<torch::Device,
                                                 torch::ScalarType,
                                                 int64_t /*batch_size*/,
                                                 int64_t /*block_size*/,
                                                 int64_t /*q_max_seq_len*/,
                                                 int64_t /*kv_max_seq_len*/,
                                                 int32_t /*sliding_window*/,
                                                 int64_t /*n_heads*/,
                                                 int64_t /*n_kv_heads*/,
                                                 int64_t /*head_dim*/,
                                                 float /*sm_scale*/,
                                                 float /*logits_soft_cap*/,
                                                 bool /*alibi*/>> {};

TEST_P(AttentionDecodeTest, KVCache) {
  const auto& [device,
               dtype,
               batch_size,
               block_size,
               q_max_seq_len,
               kv_max_seq_len,
               sliding_window,
               n_heads,
               n_kv_heads,
               head_dim,
               sm_scale,
               logits_soft_cap,
               alibi] = GetParam();
  if (device.is_cuda() && !torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA not available, skipping test";
  }

  // make sure kv_max_seq_len >= q_max_seq_len
  if (kv_max_seq_len < q_max_seq_len) {
    GTEST_SKIP() << "kv_max_seq_len < q_max_seq_len";
  }
  // total number of blocks: batch_size * max_n_blocks_per_seq
  const int32_t n_blocks =
      (kv_max_seq_len + block_size - 1) / block_size * batch_size * 2;
  // assign block ids for each sequence randomly
  std::vector<int32_t> available_block_ids(n_blocks);
  for (int32_t i = 0; i < n_blocks; ++i) {
    available_block_ids[i] = i;
  }
  std::shuffle(
      available_block_ids.begin(), available_block_ids.end(), std::mt19937());

  const int32_t max_n_blocks_per_seq =
      (kv_max_seq_len + block_size - 1) / block_size;

  std::vector<int32_t> block_tables_vec;
  std::vector<int32_t> cu_block_lens_vec = {0};
  std::vector<int32_t> slot_ids_vec;

  // generate random seq lens with size in [1, q/kv_max_seq_len]
  std::vector<int32_t> q_cu_seq_lens_vec = {0};
  std::vector<int32_t> k_cu_seq_lens_vec = {0};
  int32_t n_kv_tokens = 0;
  int32_t n_q_tokens = 0;
  absl::BitGen gen;
  for (int i = 0; i < batch_size; ++i) {
    // q_len: [1, q_max_seq_len]
    const int32_t q_len =
        absl::Uniform<int>(absl::IntervalClosedClosed, gen, 1, q_max_seq_len);
    // const int32_t q_len = q_max_seq_len;
    n_q_tokens += q_len;
    q_cu_seq_lens_vec.push_back(n_q_tokens);

    // kv_len >= q_len
    int32_t kv_len = q_len;
    if (q_len < kv_max_seq_len) {
      // sample kv_len from [q_len, kv_max_seq_len]
      kv_len = absl::Uniform<int>(
          absl::IntervalClosedClosed, gen, q_len, kv_max_seq_len);
    }
    // const int32_t kv_len = kv_max_seq_len;
    n_kv_tokens += kv_len;
    k_cu_seq_lens_vec.push_back(n_kv_tokens);

    // assign blocks for each sequence
    const int32_t n_blocks_per_seq = (kv_len + block_size - 1) / block_size;
    std::vector<int32_t> block_table;
    block_table.reserve(n_blocks_per_seq);
    for (int j = 0; j < n_blocks_per_seq; ++j) {
      ASSERT_FALSE(available_block_ids.empty());
      const int32_t block_id = available_block_ids.back();
      available_block_ids.pop_back();
      // put first slot id of each block into block_table
      block_table.push_back(block_id * block_size);
    }
    block_tables_vec.insert(
        block_tables_vec.end(), block_table.begin(), block_table.end());
    cu_block_lens_vec.push_back(static_cast<int32_t>(block_tables_vec.size()));

    // assign slots for each sequence
    for (int j = 0; j < kv_len; ++j) {
      const int32_t slot_base = block_table[j / block_size];
      const int32_t block_offset = j % block_size;
      slot_ids_vec.push_back(slot_base + block_offset);
    }
  }

  ASSERT_EQ(cu_block_lens_vec.size(), batch_size + 1);
  ASSERT_EQ(slot_ids_vec.size(), n_kv_tokens);

  // allocate memory for input tensors
  const auto options = torch::dtype(dtype).device(device);

  // generate query, key and value
  torch::Tensor query = torch::rand({n_q_tokens, n_heads, head_dim}, options);
  torch::Tensor key = torch::rand({n_kv_tokens, n_kv_heads, head_dim}, options);
  torch::Tensor value =
      torch::rand({n_kv_tokens, n_kv_heads, head_dim}, options);

  // construct key and value cache
  KVCache kv_cache(n_blocks, block_size, n_kv_heads, head_dim, options);

  // set key and value into cache
  kv_cache.set_kv_cache(slot_ids_vec, key, value);

  torch::Tensor q_cu_seq_lens = torch::tensor(
      q_cu_seq_lens_vec, torch::dtype(torch::kInt32).device(device));
  torch::Tensor k_cu_seq_lens = torch::tensor(
      k_cu_seq_lens_vec, torch::dtype(torch::kInt32).device(device));

  auto slot_ids =
      torch::tensor(slot_ids_vec, torch::dtype(torch::kInt32).device(device));
  auto block_tables = torch::tensor(block_tables_vec,
                                    torch::dtype(torch::kInt32).device(device));
  auto cu_block_lens = torch::tensor(
      cu_block_lens_vec, torch::dtype(torch::kInt32).device(device));

  torch::optional<torch::Tensor> alibi_slopes;
  if (alibi) {
    alibi_slopes =
        torch::rand({n_heads}, torch::dtype(torch::kFloat32).device(device));
  }

  InputParameters input_params;
  input_params.q_cu_seq_lens = q_cu_seq_lens;
  input_params.kv_cu_seq_lens = k_cu_seq_lens;
  input_params.q_max_seq_len = q_max_seq_len;
  input_params.kv_max_seq_len = kv_max_seq_len;
  input_params.new_cache_slots = slot_ids;
  input_params.block_tables = block_tables;
  input_params.cu_block_lens = cu_block_lens;

  RefHandler ref_handler(sm_scale, logits_soft_cap, alibi_slopes);
  torch::Tensor ref_output = torch::empty_like(query);
  // TODO: use batch_decode instead of batch_prefill
  ref_handler.batch_decode(
      query, kv_cache, input_params, sliding_window, ref_output);

  // attn handler
  ScaleAttnHandler attn_handler(sm_scale, logits_soft_cap, alibi_slopes);
  torch::Tensor output = torch::empty_like(query);
  attn_handler.batch_decode(
      query, kv_cache, input_params, sliding_window, output);

  const bool success =
      torch::allclose(ref_output, output, /*rtol=*/1e-2, /*atol=*/1e-3);
  if (!success) {
    std::cerr << "max diff: " << (ref_output - output).abs().max() << std::endl;
  }
  EXPECT_TRUE(success);
}

INSTANTIATE_TEST_SUITE_P(
    KVCache,
    AttentionDecodeTest,
    ::testing::Combine(
        ::testing::Values(torch::kCUDA),
        ::testing::Values(torch::kHalf, torch::kBFloat16),
        ::testing::Values(1, 10),                            // batch_size
        ::testing::Values(16, 64, 256),                      // block_size
        ::testing::Values(1, 10),                            // q_max_seq_len
        ::testing::Values(100, 200),                         // kv_max_seq_len
        ::testing::Values(-1, 50),                           // sliding_window
        ::testing::Values(6),                                // n_heads
        ::testing::Values(6 /*mha*/, 3 /*gqa*/, 1 /*mqa*/),  // n_kv_heads
        ::testing::Values(32, 40, 64, 128),                  // head_dim
        ::testing::Values(0.9, 1.0),                         // sm_scale
        ::testing::Values(0.0, 50.0),                        // logits_soft_cap
        ::testing::Values(false, true)                       // alibi
        ));

}  // namespace llm
