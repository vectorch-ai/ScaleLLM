#include "multi_parallel_linear.h"

#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstddef>
#include <torch/csrc/distributed/c10d/FileStore.hpp>
#include <torch/csrc/distributed/c10d/HashStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>

#include "model_loader/state_dict.h"

namespace llm {

TEST(MultiParallelLinearTest, FusedColumnParallelLinear) {
  // test load state dict for linear
  const int64_t in_features = 10;
  const int64_t out_features = 40;

  torch::Device device(torch::kCPU);
  torch::ScalarType dtype(torch::kFloat);
  const auto options = torch::dtype(dtype).device(device);

  std::vector<int64_t> out_features_vec = {
      out_features, out_features, out_features};
  std::vector<std::string> prefixes = {"query.", "key.", "value."};

  std::unordered_map<std::string, torch::Tensor> state_dict_data;
  // Allocate transposed weight matrix
  state_dict_data["query.weight"] = torch::randn({out_features, in_features});
  state_dict_data["key.weight"] = torch::randn({out_features, in_features});
  state_dict_data["value.weight"] = torch::randn({out_features, in_features});

  // weight is not sharded
  StateDict state_dict(state_dict_data);

  // test load weight
  {
    ParallelArgs parallel_args(0, 1, nullptr);
    FusedColumnParallelLinearImpl linear(in_features,
                                         out_features_vec,
                                         prefixes,
                                         /*bias=*/false,
                                         /*gather_output=*/false,
                                         parallel_args,
                                         options);
    // test load fused weight
    EXPECT_EQ(linear.load(state_dict), 3);

    for (const auto& prefix : prefixes) {
      auto named_parameters = linear.named_parameters(/*recurse=*/false);
      const auto key = detail::join_name(prefix, "weight");
      ASSERT_TRUE(named_parameters.contains(key));

      const auto& loaded_weight = named_parameters[key];
      EXPECT_EQ(loaded_weight.sizes(),
                torch::IntArrayRef({out_features, in_features}));
      EXPECT_TRUE(torch::equal(loaded_weight, state_dict_data[key]));
    }

    // verify the fused weight
    const auto loaded_fused_weight = linear.weight();
    const auto desired_fused_weight =
        torch::cat({state_dict_data["query.weight"],
                    state_dict_data["key.weight"],
                    state_dict_data["value.weight"]},
                   /*dim=*/0);
    EXPECT_TRUE(torch::equal(loaded_fused_weight, desired_fused_weight));
  }

  // test load weight with 4 shards
  const int32_t num_shards = 4;
  for (int32_t shard_id = 0; shard_id < num_shards; ++shard_id) {
    ParallelArgs parallel_args_0(shard_id, num_shards, nullptr);
    FusedColumnParallelLinearImpl linear(in_features,
                                         out_features_vec,
                                         prefixes,
                                         /*bias=*/false,
                                         /*gather_output=*/false,
                                         parallel_args_0,
                                         options);
    EXPECT_EQ(linear.load(state_dict), 3);

    auto named_parameters = linear.named_parameters(/*recurse=*/false);

    // check size for each prefix
    for (const auto& prefix : prefixes) {
      const auto key = detail::join_name(prefix, "weight");
      ASSERT_TRUE(named_parameters.contains(key));

      const auto& loaded_weight = named_parameters[key];
      EXPECT_EQ(loaded_weight.sizes(),
                torch::IntArrayRef({out_features / num_shards, in_features}));
      EXPECT_TRUE(torch::equal(
          loaded_weight, state_dict_data[key].chunk(num_shards, 0)[shard_id]));
    }

    // shard weight then cat
    auto sharded_query_weight =
        state_dict_data["query.weight"].chunk(num_shards, 0)[shard_id];
    auto sharded_key_weight =
        state_dict_data["key.weight"].chunk(num_shards, 0)[shard_id];
    auto sharded_value_weight =
        state_dict_data["value.weight"].chunk(num_shards, 0)[shard_id];

    // verify the fused weight
    const auto loaded_fused_weight = linear.weight();
    auto desired_fused_weight = torch::cat(
        {sharded_query_weight, sharded_key_weight, sharded_value_weight},
        /*dim=*/0);

    EXPECT_TRUE(torch::equal(loaded_fused_weight, desired_fused_weight));
  }
}

TEST(MultiParallelLinearTest, GroupedColumnParallelLinear) {
  const int64_t in_features = 10;
  const int64_t out_features = 40;
  std::vector<int64_t> out_features_vec = {
      out_features, out_features, out_features};
  std::vector<std::string> prefixes = {"query.", "key.", "value."};

  torch::Device device(torch::kCPU);
  torch::ScalarType dtype(torch::kFloat);
  const auto options = torch::dtype(dtype).device(device);

  std::unordered_map<std::string, torch::Tensor> state_dict_data;
  // Allocate transposed weight matrix
  state_dict_data["query.weight"] = torch::randn({out_features, in_features});
  state_dict_data["key.weight"] = torch::randn({out_features, in_features});
  state_dict_data["value.weight"] = torch::randn({out_features, in_features});
  // weight is not sharded
  StateDict state_dict(state_dict_data);

  // test load weight
  {
    ParallelArgs parallel_args(0, 1, nullptr);
    GroupedColumnParallelLinearImpl linear(in_features,
                                           out_features_vec,
                                           prefixes,
                                           /*bias=*/false,
                                           /*gather_output=*/false,
                                           parallel_args,
                                           options);
    // test load grouped weight
    EXPECT_EQ(linear.load(state_dict), 3);

    auto named_parameters = linear.named_parameters(/*recurse=*/true);
    for (size_t i = 0; i < prefixes.size(); ++i) {
      const auto prefix = "linear_" + std::to_string(i) + "." + prefixes[i];
      const auto key = detail::join_name(prefix, "weight");
      ASSERT_TRUE(named_parameters.contains(key));

      const auto& loaded_weight = named_parameters[key];

      const auto sd_key = detail::join_name(prefixes[i], "weight");

      EXPECT_EQ(loaded_weight.sizes(),
                torch::IntArrayRef({out_features, in_features}));
      EXPECT_TRUE(torch::equal(loaded_weight, state_dict_data[sd_key]));
    }
  }

  // test load weight with 4 shards
  const int32_t num_shards = 4;
  for (int32_t shard_id = 0; shard_id < num_shards; ++shard_id) {
    ParallelArgs parallel_args(shard_id, num_shards, nullptr);
    GroupedColumnParallelLinearImpl linear(in_features,
                                           out_features_vec,
                                           prefixes,
                                           /*bias=*/false,
                                           /*gather_output=*/false,
                                           parallel_args,
                                           options);
    EXPECT_EQ(linear.load(state_dict), 3);
    auto named_parameters = linear.named_parameters(/*recurse=*/true);
    // check size for each prefix
    for (size_t i = 0; i < prefixes.size(); ++i) {
      const auto prefix = "linear_" + std::to_string(i) + "." + prefixes[i];
      const auto key = detail::join_name(prefix, "weight");
      ASSERT_TRUE(named_parameters.contains(key));

      const auto& loaded_weight = named_parameters[key];
      EXPECT_EQ(loaded_weight.sizes(),
                torch::IntArrayRef({out_features / num_shards, in_features}));
      const auto sd_key = detail::join_name(prefixes[i], "weight");
      EXPECT_TRUE(
          torch::equal(loaded_weight,
                       state_dict_data[sd_key].chunk(num_shards, 0)[shard_id]));
    }
  }
}

}  // namespace llm
