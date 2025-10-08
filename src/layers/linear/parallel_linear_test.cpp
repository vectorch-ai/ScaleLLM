#include "parallel_linear.h"

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

TEST(ParallelLinearTest, RowParallelLinear) {
  // test load state dict for row parallel linear
  const int64_t in_features = 10;
  const int64_t out_features = 20;

  torch::Device device(torch::kCPU);
  torch::ScalarType dtype(torch::kFloat);
  const auto options = torch::dtype(dtype).device(device);

  std::unordered_map<std::string, torch::Tensor> state_dict_data;
  // weight is transposed
  state_dict_data["weight"] = torch::randn({out_features, in_features});
  state_dict_data["bias"] = torch::randn({out_features});
  // weight is not sharded
  StateDict state_dict(state_dict_data);
  EXPECT_EQ(state_dict_data["weight"].data_ptr(),
            state_dict.get_tensor("weight").data_ptr());

  // test load weight without sharding
  ParallelArgs parallel_args(0, 1, nullptr);
  RowParallelLinearImpl linear(in_features,
                               out_features,
                               /*bias=*/true,
                               /*input_is_parallelized=*/true,
                               parallel_args,
                               options);
  // test load state dict for transformer
  EXPECT_EQ(linear.load(state_dict), 2);
  auto named_parameters = linear.named_parameters(/*recurse=*/false);
  EXPECT_TRUE(torch::equal(state_dict.get_tensor("weight"),
                           named_parameters["weight"]));

  EXPECT_TRUE(
      torch::equal(state_dict.get_tensor("bias"), named_parameters["bias"]));

  // test load weight with 2 shards
  const int32_t num_shards = 2;
  for (int32_t shard_id = 0; shard_id < num_shards; ++shard_id) {
    ParallelArgs parallel_args(shard_id, num_shards, nullptr);
    RowParallelLinearImpl linear(in_features,
                                 out_features,
                                 /*bias=*/true,
                                 /*input_is_parallelized=*/false,
                                 parallel_args,
                                 options);
    EXPECT_EQ(linear.load(state_dict), 2);

    auto named_parameters = linear.named_parameters(/*recurse=*/false);

    const auto loaded_weight = named_parameters["weight"];

    EXPECT_EQ(loaded_weight.sizes(),
              torch::IntArrayRef({out_features, in_features / num_shards}));

    auto desired_weight = state_dict_data["weight"].chunk(/*chunks=*/num_shards,
                                                          /*dim=*/1)[shard_id];

    EXPECT_TRUE(torch::equal(loaded_weight, desired_weight));

    const auto loaded_bias = named_parameters["bias"];
    auto desired_bias = state_dict_data["bias"];
    EXPECT_TRUE(torch::equal(loaded_bias, desired_bias));
  }
}

TEST(ParallelLinearTest, ColumnParallelLinear) {
  // test load state dict for linear
  const int64_t in_features = 10;
  const int64_t out_features = 20;

  torch::Device device(torch::kCPU);
  torch::ScalarType dtype(torch::kFloat);
  const auto options = torch::dtype(dtype).device(device);

  std::unordered_map<std::string, torch::Tensor> state_dict_data;
  // Allocate transposed weight matrix
  state_dict_data["weight"] = torch::randn({out_features, in_features});
  // weight is not sharded
  StateDict state_dict(state_dict_data);
  EXPECT_EQ(state_dict_data["weight"].data_ptr(),
            state_dict.get_tensor("weight").data_ptr());

  // test load weight without sharding
  ParallelArgs parallel_args(0, 1, nullptr);
  ColumnParallelLinearImpl linear(in_features,
                                  out_features,
                                  /*bias=*/false,
                                  /*gather_output=*/false,
                                  parallel_args,
                                  options);
  // test load state dict for transformer
  EXPECT_EQ(linear.load(state_dict), 1);
  auto named_parameters = linear.named_parameters(/*recurse=*/false);
  EXPECT_TRUE(torch::equal(state_dict.get_tensor("weight"),
                           named_parameters["weight"]));

  // test load weight with 2 shards
  const int32_t num_shards = 2;
  for (int32_t shard_id = 0; shard_id < num_shards; ++shard_id) {
    ParallelArgs parallel_args(shard_id, num_shards, nullptr);
    ColumnParallelLinearImpl linear(in_features,
                                    out_features,
                                    /*bias=*/false,
                                    /*gather_output=*/false,
                                    parallel_args,
                                    options);
    EXPECT_EQ(linear.load(state_dict), 1);

    auto named_parameters = linear.named_parameters(/*recurse=*/false);

    const auto loaded_weight = named_parameters["weight"];

    EXPECT_EQ(loaded_weight.sizes(),
              torch::IntArrayRef({out_features / num_shards, in_features}));

    auto desired_weight = state_dict_data["weight"].chunk(/*chunks=*/num_shards,
                                                          /*dim=*/0)[shard_id];

    EXPECT_TRUE(torch::equal(loaded_weight, desired_weight));
  }
}

}  // namespace llm
