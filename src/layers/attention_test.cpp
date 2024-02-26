#include "attention.h"

#include <torch/torch.h>
#include <torch/types.h>

#include <cstdint>

#include "common/logging.h"
#include "gtest/gtest.h"

namespace llm {
using torch::indexing::Slice;

// helper functions to get and set key-value cache based on slot_ids
void set_kv_cache(const std::vector<int>& slot_ids,
                  const torch::Tensor& keys,
                  const torch::Tensor& values,
                  torch::Tensor& key_cache,
                  torch::Tensor& value_cache) {
  const auto num_tokens = keys.size(0);
  GCHECK(slot_ids.size() == num_tokens);

  const int64_t block_size = 256;

  // set key and value into cache one by one
  for (int64_t i = 0; i < num_tokens; ++i) {
    const int32_t slot_id = slot_ids[i];
    const auto block_id = slot_id / block_size;
    const auto block_offset = slot_id % block_size;

    // [block_id, block_offset, n_kv_heads, head_dim]
    key_cache.index_put_({block_id, block_offset, Slice(), Slice()}, keys[i]);
    value_cache.index_put_({block_id, Slice(), Slice(), block_offset},
                           values[i]);
  }
}

std::tuple<torch::Tensor, torch::Tensor> get_kv_cache(
    const std::vector<int>& slot_ids,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache) {
  const int64_t block_size = 256;

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

// Test for attention without key-value cache
class AttentionTest
    : public ::testing::TestWithParam<std::tuple<torch::Device,
                                                 torch::ScalarType,
                                                 int64_t /*batch_size*/,
                                                 int64_t /*max_seq_len*/,
                                                 int64_t /*n_heads*/,
                                                 int64_t /*n_kv_heads*/,
                                                 int64_t /*head_dim*/>> {};

TEST_P(AttentionTest, Varlen) {
  const auto& [device,
               dtype,
               batch_size,
               max_seq_len,
               n_heads,
               n_kv_heads,
               head_dim] = GetParam();

  // set random seed
  std::srand(0);

  // generate random seq lens with size in [1, max_seq_len]
  std::vector<int32_t> cu_seq_lens_vec = {0};
  int32_t n_tokens = 0;
  for (int i = 0; i < batch_size; ++i) {
    const int32_t len = std::rand() % (max_seq_len) + 1;
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

  torch::Tensor alibi_slopes =
      torch::rand({n_heads}, torch::dtype(torch::kFloat32).device(device));

  torch::Tensor output = torch::empty_like(query);
  detail::varlen_masked_self_attention_generic(query,
                                               key,
                                               value,
                                               cu_seq_lens,
                                               alibi_slopes,
                                               /*scale=*/1.0,
                                               output);

  torch::Tensor output_cuda = torch::empty_like(query);
  detail::varlen_masked_self_attention_cuda(query,
                                            key,
                                            value,
                                            cu_seq_lens,
                                            alibi_slopes,
                                            max_seq_len,
                                            /*scale=*/1.0,
                                            output_cuda);
  EXPECT_TRUE(
      torch::allclose(output, output_cuda, /*rtol=*/1e-1, /*atol=*/1e-1));
}

INSTANTIATE_TEST_SUITE_P(
    Varlen,
    AttentionTest,
    ::testing::Combine(
        ::testing::Values(torch::kCUDA),
        ::testing::Values(torch::kHalf, torch::kBFloat16),
        ::testing::Values(2, 3, 5),                           // batch_size
        ::testing::Values(200),                               // max_seq_len
        ::testing::Values(6),                                 // n_heads
        ::testing::Values(6, 3, 1),                           // n_kv_heads
        ::testing::Values(32, 40, 64, 111, 128, 224, 256)));  // head_dim

// Test for attention with key-value cache
class AttentionWithKVCacheTest
    : public ::testing::TestWithParam<
          std::tuple<torch::Device, torch::ScalarType, int64_t /*bach_size*/>> {
};

TEST_P(AttentionWithKVCacheTest, VarlenWithKVCache) {
  const auto& [device, dtype, batch_size] = GetParam();
}

INSTANTIATE_TEST_SUITE_P(
    VarlenWithKVCache,
    AttentionWithKVCacheTest,
    ::testing::Combine(::testing::Values(torch::kCUDA),
                       ::testing::Values(torch::kHalf, torch::kBFloat16),
                       ::testing::Values(256, 1088)));  // batch_size

}  // namespace llm
