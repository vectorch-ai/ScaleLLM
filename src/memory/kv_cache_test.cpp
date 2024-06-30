#include "kv_cache.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

namespace llm {

TEST(KVCacheTest, Empty) {
  KVCache kv_cache;
  EXPECT_TRUE(kv_cache.empty());
  auto [kcache, vcache] = kv_cache.get_kv_cache();
  EXPECT_FALSE(kcache.defined());
  EXPECT_FALSE(vcache.defined());
}

TEST(KVCacheTest, Basic) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA not available, skipping test";
  }

  const int num_kv_heads = 32;
  const int head_dim = 128;
  const int block_size = 8;
  const int num_blocks = 17;

  // auto dtype = torch::kFloat16;
  torch::set_default_dtype(
      torch::scalarTypeToTypeMeta(torch::ScalarType::BFloat16));
  torch::Device device(torch::kCUDA);

  torch::Tensor key_cache =
      torch::rand({num_blocks, block_size, num_kv_heads, head_dim},
                  /*device=*/device);
  torch::Tensor value_cache =
      torch::rand({num_blocks, block_size, num_kv_heads, head_dim},
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
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA not available, skipping test";
  }
  
  const int64_t num_kv_heads = 12;
  const int64_t head_dim = 128;
  const int64_t block_size = 4;
  const int64_t num_blocks = 2;

  // auto dtype = torch::kFloat16;
  torch::set_default_dtype(
      torch::scalarTypeToTypeMeta(torch::ScalarType::BFloat16));
  torch::Device device(torch::kCUDA);

  torch::manual_seed(10);

  torch::Tensor key_cache =
      torch::rand({num_blocks, block_size, num_kv_heads, head_dim},
                  /*device=*/device);
  torch::Tensor value_cache =
      torch::rand({num_blocks, block_size, num_kv_heads, head_dim},
                  /*device=*/device);

  KVCache kv_cache(key_cache, value_cache);

  for (int32_t i = 0; i < 10000; ++i) {
    using ISlice = torch::indexing::Slice;

    const int64_t sample_size = std::min<int64_t>(num_blocks * block_size, 10);
    const int64_t num_slots = i % sample_size + 1;
    torch::Tensor slot_ids =
        torch::randperm(num_blocks * block_size,
                        torch::dtype(torch::kInt).device(device))
            .index({ISlice(0, num_slots)});

    // construct keys and values with different strides
    torch::Tensor keys =
        torch::rand({num_slots, num_kv_heads * 2, head_dim},
                    torch::device(device))
            .slice(/*dim=*/1, /*start=*/0, /*end=*/num_kv_heads);
    torch::Tensor values =
        torch::rand({num_slots, num_kv_heads, head_dim}, torch::device(device));
    EXPECT_NE(keys.stride(0), values.stride(0));

    kv_cache.set_kv_cache_cuda(slot_ids, keys, values);

    auto [keys_out, values_out] = kv_cache.get_kv_cache(slot_ids);
    ASSERT_TRUE(torch::equal(keys, keys_out));
    ASSERT_TRUE(torch::equal(values, values_out));
  }
}

}  // namespace llm
