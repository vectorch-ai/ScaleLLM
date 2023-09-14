#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <memory>
#include <thread>
#include <torch/csrc/distributed/c10d/FileStore.hpp>
#include <torch/csrc/distributed/c10d/HashStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>

#include "linear.h"
#include "torch_utils/state_dict.h"

namespace llm {


TEST(LayersTest, TestLoadStateDict) {
  // test load state dict for linear
  const int64_t in_features = 10;
  const int64_t out_features = 20;

  torch::Device device(torch::kCPU);
  torch::ScalarType dtype(torch::kFloat);
  ParallelArgs parallel_args(0, 1, nullptr);
  ColumnParallelLinear linear(in_features,
                              out_features,
                              /*gather_output=*/false,
                              parallel_args,
                              dtype,
                              device);
  std::unordered_map<std::string, torch::Tensor> state_dict_data;
  // Allocate transposed weight matrix
  state_dict_data["weight"] = torch::randn({out_features, in_features});

  StateDict state_dict(state_dict_data);
  // test load state dict for transformer
  linear->load_state_dict(state_dict);

  EXPECT_EQ(state_dict_data["weight"].data_ptr(),
            state_dict.get_tensor("weight").data_ptr());

  auto named_parameters = linear->named_parameters(/*recurse=*/false);
  EXPECT_TRUE(
      torch::equal(state_dict_data["weight"], named_parameters["weight"]));
}

}  // namespace llm
