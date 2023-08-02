#include "linear.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include "torch_utils/state_dict.h"

namespace llm {

TEST(LayersTest, TestLoadStateDict) {
  // test load state dict for linear
  const int64_t in_features = 10;
  const int64_t out_features = 20;

  ColumnParallelLinear linear(in_features, out_features, 1);
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
