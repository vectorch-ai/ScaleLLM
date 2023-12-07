#include "kv_cache.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

namespace llm {

TEST(KVCacheTest, Basic) {
  const int num_kv_heads = 32;
  const int head_dim = 128;
  const int block_size = 8;
  const int x = 8;
  const int num_blocks = 17;

  // auto dtype = torch::kFloat16;
  torch::set_default_dtype(
      torch::scalarTypeToTypeMeta(torch::ScalarType::BFloat16));
  torch::Device device(torch::kCUDA);

  torch::Tensor key_cache =
      torch::rand({num_blocks, num_kv_heads, head_dim / x, block_size, x},
                  /*device=*/device);
  torch::Tensor value_cache =
      torch::rand({num_blocks, num_kv_heads, head_dim, block_size},
                  /*device=*/device);

  KVCache kv_cache(key_cache, value_cache);

  // set key and value cache for the given slot_ids
  for (int32_t i = 0; i < num_blocks * block_size; ++i) {
    torch::Tensor slot_ids =
        torch::tensor({i}, torch::dtype(torch::kInt).device(device));
    torch::Tensor keys =
        torch::ones({1, num_kv_heads, head_dim}, /*device=*/device) * i;
    torch::Tensor values =
        torch::ones({1, num_kv_heads, head_dim}, /*device=*/device) * i;
    kv_cache.set_kv_cache_cuda(slot_ids, keys, values);
  }

  // get key and value cache for the given slot_ids
  for (int32_t i = 0; i < num_blocks * block_size; ++i) {
    torch::Tensor slot_ids =
        torch::tensor({i}, torch::dtype(torch::kInt).device(device));
    auto [keys, values] = kv_cache.get_kv_cache(slot_ids);
    auto desired =
        torch::ones({1, num_kv_heads, head_dim}, /*device=*/device) * i;
    EXPECT_TRUE(torch::equal(keys, desired));
    EXPECT_TRUE(torch::equal(values, desired));
  }
}

TEST(KVCacheTest, Random) {
  const int num_kv_heads = 12;
  const int head_dim = 128;
  const int block_size = 4;
  const int x = 8;
  const int num_blocks = 2;

  // auto dtype = torch::kFloat16;
  torch::set_default_dtype(
      torch::scalarTypeToTypeMeta(torch::ScalarType::BFloat16));
  torch::Device device(torch::kCUDA);

  torch::manual_seed(10);

  torch::Tensor key_cache =
      torch::rand({num_blocks, num_kv_heads, head_dim / x, block_size, x},
                  /*device=*/device);
  torch::Tensor value_cache =
      torch::rand({num_blocks, num_kv_heads, head_dim, block_size},
                  /*device=*/device);

  KVCache kv_cache(key_cache, value_cache);

  for (int32_t i = 0; i < 10000; ++i) {
    using torch::indexing::Slice;

    const int sample_size = std::min(num_blocks * block_size, 10);
    const int num_slots = i % sample_size + 1;
    torch::Tensor slot_ids =
        torch::randperm(num_blocks * block_size,
                        torch::dtype(torch::kInt).device(device))
            .index({Slice(0, num_slots)});

    torch::Tensor keys =
        torch::rand({num_slots, num_kv_heads, head_dim}, torch::device(device));
    torch::Tensor values =
        torch::rand({num_slots, num_kv_heads, head_dim}, torch::device(device));

    kv_cache.set_kv_cache_cuda(slot_ids, keys, values);

    auto [keys_out, values_out] = kv_cache.get_kv_cache(slot_ids);
    ASSERT_TRUE(torch::equal(keys, keys_out));
    ASSERT_TRUE(torch::equal(values, values_out));
  }
}

}  // namespace llm
