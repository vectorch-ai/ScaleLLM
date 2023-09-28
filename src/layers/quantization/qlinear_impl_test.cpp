#include "qlinear_gptq_impl.h"

#include <c10/core/TensorOptions.h>
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

TEST(QlinearTest, ColumnParallelQuantLinear) {
  const int64_t in_features = 4096;
  const int64_t out_features = 4096;
  const int64_t bits = 4;
  const int64_t group_size = 128;
  ColumnParallelQLinearGPTQImpl qlinear(in_features,
                                        out_features,
                                        /*bias=*/false,
                                        bits,
                                        group_size,
                                        /*gather_output=*/false,
                                        ParallelArgs(0, 1, nullptr),
                                        /*dtype=*/torch::kHalf,
                                        /*device=*/torch::kCUDA);
  auto state_dict = StateDict::load_safetensors(
      "data/gptq.safetensors", /*shard_id=*/0, /*num_shards=*/1);
  auto weights = details::construct_weights(state_dict->get_tensor("qweight"),
                                            state_dict->get_tensor("qzeros"),
                                            state_dict->get_tensor("scales"),
                                            /*bits=*/4);
  weights = weights.to(torch::kCUDA);

  qlinear.load_state_dict(*state_dict);
  qlinear.verify_loaded_weights();

  auto input = torch::rand({40960, in_features},
                           torch::dtype(torch::kHalf).device(torch::kCUDA));
  auto output = qlinear.forward(input);
  auto desired_output = torch::matmul(input, weights);
  EXPECT_TRUE(torch::allclose(output,
                              desired_output,
                              /*rtol=*/1e-01,
                              /*atol=*/1e-02));
}

TEST(QlinearTest, RowParallelQuantLinear) {
  const int64_t in_features = 4096;
  const int64_t out_features = 4096;
  const int64_t bits = 4;
  const int64_t group_size = 128;
  RowParallelQLinearGPTQImpl qlinear(in_features,
                                     out_features,
                                     /*bias=*/false,
                                     bits,
                                     group_size,
                                     /*input_is_parallelized=*/true,
                                     ParallelArgs(0, 1, nullptr),
                                     /*dtype=*/torch::kHalf,
                                     /*device=*/torch::kCUDA);
  auto state_dict = StateDict::load_safetensors(
      "data/gptq.safetensors", /*shard_id=*/0, /*num_shards=*/1);
  auto weights = details::construct_weights(state_dict->get_tensor("qweight"),
                                            state_dict->get_tensor("qzeros"),
                                            state_dict->get_tensor("scales"),
                                            /*bits=*/4);
  weights = weights.to(torch::kCUDA);

  qlinear.load_state_dict(*state_dict);
  qlinear.verify_loaded_weights();

  auto input = torch::rand({40960, in_features},
                           torch::dtype(torch::kHalf).device(torch::kCUDA));
  auto output = qlinear.forward(input);
  auto desired_output = torch::matmul(input, weights);
  EXPECT_TRUE(torch::allclose(output,
                              desired_output,
                              /*rtol=*/1e-01,
                              /*atol=*/1e-02));
}

}  // namespace llm
