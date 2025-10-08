#include "parallel_qlinear_gptq.h"

#include <c10/core/TensorOptions.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include "model_loader/state_dict.h"

namespace llm {

TEST(GPTQQlinearTest, ColumnParallelQLinear) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA not available, skipping test";
  }

  const int64_t in_features = 4096;
  const int64_t out_features = 4096;
  QuantArgs quant_args;
  quant_args.bits(4);
  quant_args.group_size(128);
  const auto options = torch::dtype(torch::kHalf).device(torch::kCUDA);
  ColumnParallelQLinearGPTQImpl qlinear(in_features,
                                        out_features,
                                        /*bias=*/false,
                                        quant_args,
                                        /*gather_output=*/false,
                                        ParallelArgs(0, 1, nullptr),
                                        options);
  auto state_dict = StateDict::load_safetensors("data/gptq.safetensors");
  auto weights = detail::construct_weights(state_dict->get_tensor("qweight"),
                                           state_dict->get_tensor("qzeros"),
                                           state_dict->get_tensor("scales"),
                                           /*bits=*/4);
  weights = weights.to(torch::kCUDA);

  qlinear.load(*state_dict);
  EXPECT_TRUE(qlinear.verify());

  auto input = torch::rand({40960, in_features}, options);
  auto output = qlinear.forward(input);
  auto desired_output = torch::matmul(input, weights);
  EXPECT_TRUE(torch::allclose(output,
                              desired_output,
                              /*rtol=*/1e-01,
                              /*atol=*/1e-02));
}

TEST(GPTQQlinearTest, RowParallelQLinear) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA not available, skipping test";
  }

  const int64_t in_features = 4096;
  const int64_t out_features = 4096;
  QuantArgs quant_args;
  quant_args.bits(4);
  quant_args.group_size(128);
  const auto options = torch::dtype(torch::kHalf).device(torch::kCUDA);
  RowParallelQLinearGPTQImpl qlinear(in_features,
                                     out_features,
                                     /*bias=*/false,
                                     quant_args,
                                     /*input_is_parallelized=*/true,
                                     ParallelArgs(0, 1, nullptr),
                                     options);
  auto state_dict = StateDict::load_safetensors("data/gptq.safetensors");
  auto weights = detail::construct_weights(state_dict->get_tensor("qweight"),
                                           state_dict->get_tensor("qzeros"),
                                           state_dict->get_tensor("scales"),
                                           /*bits=*/4);
  weights = weights.to(torch::kCUDA);

  qlinear.load(*state_dict);
  EXPECT_TRUE(qlinear.verify());

  auto input = torch::rand({40960, in_features}, options);
  auto output = qlinear.forward(input);
  auto desired_output = torch::matmul(input, weights);
  EXPECT_TRUE(torch::allclose(output,
                              desired_output,
                              /*rtol=*/1e-01,
                              /*atol=*/1e-02));
}

}  // namespace llm
