#include "qlinear_impl.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "model_loader/state_dict.h"

namespace llm {

TEST(QlinearTest, Basic) {
  auto state_dict = StateDict::load_safetensors(
      "data/gptq_small.safetensors", /*shard_id=*/0, /*num_shards=*/1);
  auto weights = details::construct_weights(state_dict->get_tensor("qweight"),
                                            state_dict->get_tensor("qzeros"),
                                            state_dict->get_tensor("scales"),
                                            /*bits=*/4);
  auto weights_2 = details::construct_weights(state_dict->get_tensor("qweight"),
                                              state_dict->get_tensor("qzeros"),
                                              state_dict->get_tensor("scales"),
                                              state_dict->get_tensor("g_idx"),
                                              /*bits=*/4);
  EXPECT_TRUE(torch::allclose(weights, weights_2));
}

}  // namespace llm
