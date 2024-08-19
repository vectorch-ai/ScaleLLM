#include "qkv_linear.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <tuple>

#include "model_loader/state_dict.h"

namespace llm {

class QKVLinearTest
    : public ::testing::TestWithParam<std::tuple<int64_t /*n_tokens*/,
                                                 int64_t /*n_heads*/,
                                                 int64_t /*n_kv_heads*/,
                                                 int64_t /*n_shards*/,
                                                 int64_t /*head_dim*/,
                                                 int64_t /*hidden_size*/>> {};

TEST_P(QKVLinearTest, LoadFusedWeight) {
  const auto& [n_tokens, n_heads, n_kv_heads, n_shards, head_dim, hidden_size] =
      GetParam();

  ASSERT_LE(n_kv_heads, n_heads);
  ASSERT_LE(n_shards, n_heads);

  torch::Device device(torch::kCPU);
  torch::ScalarType dtype(torch::kFloat);
  const auto options = torch::dtype(dtype).device(device);
  std::unordered_map<std::string, torch::Tensor> state_dict_data;
  // Allocate transposed weight matrix
  state_dict_data["query.weight"] =
      torch::randn({n_heads * head_dim, hidden_size}, options);
  state_dict_data["key.weight"] =
      torch::randn({n_kv_heads * head_dim, hidden_size}, options);
  state_dict_data["value.weight"] =
      torch::randn({n_kv_heads * head_dim, hidden_size}, options);

  // weight is not sharded
  StateDict state_dict(state_dict_data);

  int64_t n_kv_shards = std::min(n_kv_heads, n_shards);
  auto query_chunks = state_dict_data["query.weight"].chunk(
      /*chunks=*/n_shards, /*dim=*/0);
  auto key_chunks = state_dict_data["key.weight"].chunk(
      /*chunks=*/n_kv_shards, /*dim=*/0);
  auto value_chunks = state_dict_data["value.weight"].chunk(
      /*chunks=*/n_kv_shards, /*dim=*/0);

  // test load weight
  for (int32_t shard_id = 0; shard_id < n_shards; ++shard_id) {
    QuantArgs quant_args;
    ParallelArgs parallel_args(shard_id, n_shards, nullptr);
    QKVColumnParallelLinearImpl linear(hidden_size,
                                       n_heads,
                                       n_kv_heads,
                                       head_dim,
                                       /*bias=*/false,
                                       /*gather_output=*/false,
                                       quant_args,
                                       parallel_args,
                                       options);
    linear.load_state_dict(state_dict,
                           /*prefixes=*/{"query.", "key.", "value."},
                           /*kv_prefixes=*/{"key.", "value."});

    // generate random input and compare with the output
    auto input = torch::randn({n_tokens, hidden_size}, options);
    auto qkv = linear.forward(input);

    const int64_t kv_shard_id =
        n_kv_heads >= n_shards ? shard_id : n_kv_heads * shard_id / n_shards;

    auto query = input.matmul(query_chunks[shard_id].t());
    EXPECT_TRUE(torch::allclose(qkv[0], query, /*rtol=*/1e-5, /*atol=*/1e-5));

    auto key = input.matmul(key_chunks[kv_shard_id].t());
    EXPECT_TRUE(torch::allclose(qkv[1], key, /*rtol=*/1e-5, /*atol=*/1e-5));

    auto value = input.matmul(value_chunks[kv_shard_id].t());
    EXPECT_TRUE(torch::allclose(qkv[2], value, /*rtol=*/1e-5, /*atol=*/1e-5));
  }
}

INSTANTIATE_TEST_SUITE_P(
    QKVLinearTestSuite,
    QKVLinearTest,
    ::testing::Combine(::testing::Values(10, 32),      // n_tokens
                       ::testing::Values(8, 16, 32),   // n_heads
                       ::testing::Values(1, 2, 4, 8),  // n_kv_heads
                       ::testing::Values(1, 2, 4, 8),  // n_shards
                       ::testing::Values(1, 32, 64),   // head_dim
                       ::testing::Values(1, 32, 64))   // hidden_size
);

}  // namespace llm
