#include "qkv_linear.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <memory>
#include <thread>
#include <tuple>

#include "model_loader/state_dict.h"

namespace llm {

class QKVLinearTest
    : public ::testing::TestWithParam<std::tuple<int64_t /*n_heads*/,
                                                 int64_t /*n_kv_heads*/,
                                                 int32_t /*n_shards*/,
                                                 int64_t /*head_dim*/,
                                                 int64_t /*hidden_size*/>> {};

TEST_P(QKVLinearTest, LoadFusedWeight) {
  const auto& [n_heads, n_kv_heads, n_shards, head_dim, hidden_size] =
      GetParam();

  ASSERT_LE(n_kv_heads, n_heads);
  ASSERT_LE(n_shards, n_heads);

  torch::Device device(torch::kCPU);
  torch::ScalarType dtype(torch::kFloat);
  const auto options = torch::dtype(dtype).device(device);
  std::unordered_map<std::string, torch::Tensor> state_dict_data;
  // Allocate transposed weight matrix
  state_dict_data["query.weight"] =
      torch::randn({n_heads * head_dim, hidden_size});
  state_dict_data["key.weight"] =
      torch::randn({n_kv_heads * head_dim, hidden_size});
  state_dict_data["value.weight"] =
      torch::randn({n_kv_heads * head_dim, hidden_size});

  // weight is not sharded
  StateDict state_dict(state_dict_data, /*shard_id=*/0, /*num_shards=*/1);

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

    auto named_parameters = linear.named_parameters(/*recurse=*/true);

    const auto loaded_weight = named_parameters["parallel_linear.weight"];

    int32_t n_kv_shards = n_shards;
    int32_t kv_shard_id = shard_id;
    if (n_kv_heads < n_heads) {
      ASSERT_EQ(n_heads % n_kv_heads, 0);
      if (n_kv_heads < n_shards) {
        ASSERT_EQ(n_shards % n_kv_heads, 0);
        n_kv_shards = n_kv_heads;
        const int32_t replication_ratio = n_shards / n_kv_heads;
        kv_shard_id = shard_id / replication_ratio;
      }
    }

    // shard weight then cat
    auto query_weight = state_dict_data["query.weight"].chunk(
        /*chunks=*/n_shards, /*dim=*/0)[shard_id];
    auto key_weight =
        state_dict_data["key.weight"].chunk(/*chunks=*/n_kv_shards,
                                            /*dim=*/0)[kv_shard_id];
    auto value_weight = state_dict_data["value.weight"].chunk(
        /*chunks=*/n_kv_shards, /*dim=*/0)[kv_shard_id];

    auto desired_weight = torch::cat({query_weight, key_weight, value_weight},
                                     /*dim=*/0);

    EXPECT_TRUE(torch::equal(loaded_weight, desired_weight));
  }
}

INSTANTIATE_TEST_SUITE_P(
    QKVLinearTestSuite,
    QKVLinearTest,
    ::testing::Combine(::testing::Values(8, 16, 32),   // n_heads
                       ::testing::Values(1, 2, 4, 8),  // n_kv_heads
                       ::testing::Values(1, 2, 4, 8),  // n_shards
                       ::testing::Values(1, 32, 64),   // head_dim
                       ::testing::Values(1, 32, 64))   // hidden_size
);

}  // namespace llm
