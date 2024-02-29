#include "attention.h"

#include <ATen/ops/equal.h>
#include <absl/random/random.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/Optional.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/cuda.h>
#include <torch/torch.h>
#include <torch/types.h>

#include <cstdint>

#include "common/logging.h"
#include "gtest/gtest.h"

namespace llm {
using torch::indexing::Slice;

// helper functions to get and set key-value cache based on slot_ids
void set_kv_cache(
    const std::vector<int>& slot_ids,
    const torch::Tensor& keys,    // [n_tokens, n_kv_heads, head_dim]
    const torch::Tensor& values,  // [n_tokens, n_kv_heads, head_dim]
    torch::Tensor& key_cache,  // [n_blocks, block_size, n_kv_heads, head_dim]
    torch::Tensor& value_cache) {
  const auto n_tokens = keys.size(0);
  GCHECK(slot_ids.size() == n_tokens);

  // [n_blocks, block_size, n_kv_heads, head_dim]
  const int64_t block_size = key_cache.size(1);

  // set key and value into cache one by one
  for (int64_t i = 0; i < n_tokens; ++i) {
    const int32_t slot_id = slot_ids[i];
    const auto block_id = slot_id / block_size;
    const auto block_offset = slot_id % block_size;

    // [block_id, block_offset, n_kv_heads, head_dim]
    key_cache.index_put_({block_id, block_offset, Slice(), Slice()}, keys[i]);
    value_cache.index_put_({block_id, block_offset, Slice(), Slice()},
                           values[i]);
  }
}

std::tuple<torch::Tensor, torch::Tensor> get_kv_cache(
    const std::vector<int>& slot_ids,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache) {
  // [n_blocks, block_size, n_kv_heads, head_dim]
  const int64_t block_size = key_cache.size(1);

  std::vector<torch::Tensor> keys;
  std::vector<torch::Tensor> values;
  // get key and value from cache one by one
  for (int slot_id : slot_ids) {
    const auto block_id = slot_id / block_size;
    const auto block_offset = slot_id % block_size;
    // key = key_cache_[block_id, :, :, block_offset, :]
    const auto key =
        key_cache.index({block_id, block_offset, Slice(), Slice()});
    keys.push_back(key);
    // value = value_cache_[block_id, :, :, block_offset]
    const auto value =
        value_cache.index({block_id, block_offset, Slice(), Slice()});
    values.push_back(value);
  }
  return std::make_tuple(torch::stack(keys), torch::stack(values));
}

// Tests self-attention for prefill stage
class AttentionPrefillTest
    : public ::testing::TestWithParam<std::tuple<torch::Device,
                                                 torch::ScalarType,
                                                 int64_t /*batch_size*/,
                                                 int64_t /*max_seq_len*/,
                                                 int64_t /*n_heads*/,
                                                 int64_t /*n_kv_heads*/,
                                                 int64_t /*head_dim*/,
                                                 float /*scale*/,
                                                 bool /*alibi*/>> {};

TEST_P(AttentionPrefillTest, Varlen) {
  const auto& [device,
               dtype,
               batch_size,
               max_seq_len,
               n_heads,
               n_kv_heads,
               head_dim,
               scale,
               alibi] = GetParam();

  absl::BitGen gen;

  // generate random seq lens with size in [1, max_seq_len]
  std::vector<int32_t> cu_seq_lens_vec = {0};
  int32_t n_tokens = 0;
  for (int i = 0; i < batch_size; ++i) {
    const int32_t len =
        absl::Uniform<int>(absl::IntervalClosedClosed, gen, 1, max_seq_len);
    n_tokens += len;
    cu_seq_lens_vec.push_back(n_tokens);
  }

  // allocate memory for input tensors
  const auto options = torch::dtype(dtype).device(device);
  torch::Tensor query = torch::rand({n_tokens, n_heads, head_dim}, options);
  torch::Tensor key = torch::rand({n_tokens, n_kv_heads, head_dim}, options);
  torch::Tensor value = torch::rand({n_tokens, n_kv_heads, head_dim}, options);

  torch::Tensor cu_seq_lens = torch::tensor(
      cu_seq_lens_vec, torch::dtype(torch::kInt32).device(device));
  torch::Tensor none_tensor;

  torch::optional<torch::Tensor> alibi_slopes;
  if (alibi) {
    alibi_slopes =
        torch::rand({n_heads}, torch::dtype(torch::kFloat32).device(device));
  }

  torch::Tensor output = torch::empty_like(query);
  detail::varlen_masked_self_attention_generic(
      query, key, value, cu_seq_lens, alibi_slopes, scale, output);

  torch::Tensor output_cuda = torch::empty_like(query);
  detail::varlen_masked_self_attention_cuda(query,
                                            key,
                                            value,
                                            cu_seq_lens,
                                            alibi_slopes,
                                            max_seq_len,
                                            scale,
                                            output_cuda);
  EXPECT_TRUE(
      torch::allclose(output, output_cuda, /*rtol=*/1e-1, /*atol=*/1e-1));
}

INSTANTIATE_TEST_SUITE_P(
    Varlen,
    AttentionPrefillTest,
    ::testing::Combine(::testing::Values(torch::kCUDA),
                       ::testing::Values(torch::kHalf, torch::kBFloat16),
                       ::testing::Values(2, 3, 5),          // batch_size
                       ::testing::Values(200),              // max_seq_len
                       ::testing::Values(6),                // n_heads
                       ::testing::Values(6, 3, 1),          // n_kv_heads
                       ::testing::Values(32, 40, 64, 128),  // head_dim
                       ::testing::Values(0.9, 1.0),         // scale
                       ::testing::Values(false, true)       // alibi
                       ));

// Test attention with kv-cache for decode stage
class AttentionDecodeTest
    : public ::testing::TestWithParam<std::tuple<torch::Device,
                                                 torch::ScalarType,
                                                 int64_t /*batch_size*/,
                                                 int64_t /*block_size*/,
                                                 int64_t /*q_max_seq_len*/,
                                                 int64_t /*kv_max_seq_len*/,
                                                 int64_t /*n_heads*/,
                                                 int64_t /*n_kv_heads*/,
                                                 int64_t /*head_dim*/,
                                                 float /*scale*/,
                                                 bool /*alibi*/,
                                                 int32_t /*num_splits*/>> {};

TEST_P(AttentionDecodeTest, KVCache) {
  const auto& [device,
               dtype,
               batch_size,
               block_size,
               q_max_seq_len,
               kv_max_seq_len,
               n_heads,
               n_kv_heads,
               head_dim,
               scale,
               alibi,
               num_splits] = GetParam();
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

  std::vector<std::vector<int32_t>> block_tables_vec;
  std::vector<int> slot_ids;

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
    std::vector<int32_t> block_table(max_n_blocks_per_seq);
    const int32_t n_blocks_per_seq = (kv_len + block_size - 1) / block_size;
    for (int j = 0; j < n_blocks_per_seq; ++j) {
      ASSERT_FALSE(available_block_ids.empty());
      block_table[j] = available_block_ids.back();
      available_block_ids.pop_back();
    }
    block_tables_vec.push_back(block_table);

    // assign slots for each sequence
    for (int j = 0; j < kv_len; ++j) {
      const int32_t block_id = block_table[j / block_size];
      const int32_t block_offset = j % block_size;
      slot_ids.push_back(block_id * block_size + block_offset);
    }
  }

  ASSERT_EQ(block_tables_vec.size(), batch_size);
  ASSERT_EQ(slot_ids.size(), n_kv_tokens);

  // allocate memory for input tensors
  const auto options = torch::dtype(dtype).device(device);

  // generate query, key and value
  torch::Tensor query = torch::rand({n_q_tokens, n_heads, head_dim}, options);
  torch::Tensor key = torch::rand({n_kv_tokens, n_kv_heads, head_dim}, options);
  torch::Tensor value =
      torch::rand({n_kv_tokens, n_kv_heads, head_dim}, options);

  // construct key and value cache
  const std::vector<int64_t> kv_shape = {
      n_blocks, block_size, n_kv_heads, head_dim};
  torch::Tensor k_cache = torch::empty(kv_shape, options);
  torch::Tensor v_cache = torch::empty(kv_shape, options);

  // set key and value into cache based on slot_ids
  set_kv_cache(slot_ids, key, value, k_cache, v_cache);
  auto [k, v] = get_kv_cache(slot_ids, k_cache, v_cache);
  ASSERT_TRUE(torch::equal(k, key));
  ASSERT_TRUE(torch::equal(v, value));

  torch::Tensor q_cu_seq_lens = torch::tensor(
      q_cu_seq_lens_vec, torch::dtype(torch::kInt32).device(device));
  torch::Tensor k_cu_seq_lens = torch::tensor(
      k_cu_seq_lens_vec, torch::dtype(torch::kInt32).device(device));

  // construct block tables with padding=0
  auto block_tables = torch::empty(
      {static_cast<int32_t>(block_tables_vec.size()), max_n_blocks_per_seq},
      torch::dtype(torch::kInt32).device(device));
  for (int64_t i = 0; i < block_tables_vec.size(); ++i) {
    block_tables.index_put_({i, Slice()},
                            torch::tensor(block_tables_vec[i], torch::kInt));
  }

  torch::optional<torch::Tensor> alibi_slopes;
  if (alibi) {
    alibi_slopes =
        torch::rand({n_heads}, torch::dtype(torch::kFloat32).device(device));
  }

  torch::Tensor output = torch::empty_like(query);
  detail::multiple_query_masked_self_attention_cuda(
      query,
      key,
      value,
      q_cu_seq_lens,
      k_cu_seq_lens,
      /*block_tables=*/torch::nullopt,
      alibi_slopes,
      q_max_seq_len,
      kv_max_seq_len,
      scale,
      output,
      /*num_splits=*/1);

  torch::Tensor output_with_cache = torch::empty_like(query);
  detail::multiple_query_masked_self_attention_cuda(query,
                                                    k_cache,
                                                    v_cache,
                                                    q_cu_seq_lens,
                                                    k_cu_seq_lens,
                                                    block_tables,
                                                    alibi_slopes,
                                                    q_max_seq_len,
                                                    kv_max_seq_len,
                                                    scale,
                                                    output_with_cache,
                                                    num_splits);

  EXPECT_TRUE(
      torch::allclose(output_with_cache, output, /*rtol=*/1e-2, /*atol=*/1e-3));
}

INSTANTIATE_TEST_SUITE_P(
    KVCache,
    AttentionDecodeTest,
    ::testing::Combine(
        ::testing::Values(torch::kCUDA),
        ::testing::Values(torch::kHalf, torch::kBFloat16),
        ::testing::Values(1, 10),                            // batch_size
        ::testing::Values(16, 80, 256),                      // block_size
        ::testing::Values(1, 10),                            // q_max_seq_len
        ::testing::Values(100, 200),                         // kv_max_seq_len
        ::testing::Values(6),                                // n_heads
        ::testing::Values(6 /*mha*/, 3 /*gqa*/, 1 /*mqa*/),  // n_kv_heads
        ::testing::Values(32, 40, 64, 128),                  // head_dim
        ::testing::Values(0.9, 1.0),                         // scale
        ::testing::Values(false, true),                      // alibi
        ::testing::Values(0, 1)                              // num_splits
        ));

}  // namespace llm
