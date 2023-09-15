#include "state_dict.h"

#include <c10/core/Device.h>
#include <gtest/gtest.h>

namespace llm {

// test data was generated with the following python code:
// import torch
// from safetensors.torch import save_file
// tensors = {}
// for i in range(20):
//     tensors[f"key_{i}"] = torch.ones(10, 10) * i;
// save_file(tensors, "test.safetensors")
// torch.save(tensors, "test.pth")

TEST(StateDictTest, LoadPickle) {
  auto state_dict = StateDict::load_pickle_file("data/test.pth");
  EXPECT_EQ(state_dict->size(), 20);
  for (int i = 0; i < 20; ++i) {
    const std::string key = "key_" + std::to_string(i);
    auto tensor = state_dict->get_tensor(key);
    ASSERT_TRUE(tensor.defined());
    EXPECT_EQ(tensor.numel(), 100);
    EXPECT_EQ(tensor.dtype(), torch::kFloat32);
    EXPECT_EQ(tensor.sizes(), torch::IntArrayRef({10, 10}));
    EXPECT_TRUE(tensor.equal(torch::ones({10, 10}) * i));
  }
}

TEST(StateDictTest, LoadSafeTensors) {
  auto state_dict = StateDict::load_safetensors("data/test.safetensors");
  EXPECT_EQ(state_dict->size(), 20);
  for (int i = 0; i < 20; ++i) {
    const std::string key = "key_" + std::to_string(i);
    auto tensor = state_dict->get_tensor(key);
    ASSERT_TRUE(tensor.defined());
    EXPECT_EQ(tensor.numel(), 100);
    EXPECT_EQ(tensor.dtype(), torch::kFloat32);
    EXPECT_EQ(tensor.sizes(), torch::IntArrayRef({10, 10}));
    EXPECT_TRUE(tensor.equal(torch::ones({10, 10}) * i));
  }
}

}  // namespace llm
