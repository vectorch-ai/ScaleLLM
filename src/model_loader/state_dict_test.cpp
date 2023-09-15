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

TEST(StateDictTest, SharedTensor) {
  // TODO: add more tests
  // create a list of tensors with same size
  std::vector<torch::Tensor> tensors;
  for (int i = 0; i < 8; ++i) {
    tensors.push_back(torch::ones({2, 2})*i);
  }
  torch::Tensor tensor = torch::cat(tensors, /*dim=*/0);
  EXPECT_EQ(tensor.sizes(), torch::IntArrayRef({16, 2}));
  StateDict state_dict({{"tensor", tensor}});
  state_dict.set_shard(0, 1);
  EXPECT_EQ(state_dict.size(), 1);

  // test get_tensor
  auto tensor1 = state_dict.get_tensor("tensor");
  EXPECT_TRUE(tensor1.equal(tensor));

  // test get_sharded_tensor
  auto chunks = tensor.chunk(2, /*dim=*/0);
  auto rank0_tensor = state_dict.get_sharded_tensor("tensor",
                                               /*dim=*/0,
                                               /*rank=*/0,
                                               /*world_size=*/2);
  LOG(ERROR) << rank0_tensor;
  EXPECT_EQ(rank0_tensor.sizes(), torch::IntArrayRef({8, 2}));
  EXPECT_TRUE(rank0_tensor.equal(chunks[0]));
  auto rank1_tensor = state_dict.get_sharded_tensor("tensor",
                                               /*dim=*/0,
                                               /*rank=*/1,
                                               /*world_size=*/2);
  LOG(ERROR) << rank1_tensor;
  EXPECT_EQ(rank1_tensor.sizes(), torch::IntArrayRef({8, 2}));
  EXPECT_TRUE(rank1_tensor.equal(chunks[1]));
}

}  // namespace llm
