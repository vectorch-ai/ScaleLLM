#include "sampler.h"

#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorImpl.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <kernels/sampling/sampling_kernels.h>
#include <torch/torch.h>
#include <torch/types.h>

namespace llm {

TEST(SamplerTest, Greedy) {
  // Test GreedySampler
  torch::ScalarType dtype(torch::kFloat32);
  torch::Device device(torch::kCPU);
  SamplingParameters params;
  params.top_k = {0, 0};
  params.top_p = {1.0, 1.0};
  params.do_sample = {false, false};
  Sampler sampler(params, dtype, device);

  int64_t batch_size = 2;
  int64_t vocab_size = 32000;
  const auto logits = torch::randn({batch_size, vocab_size},
                                   torch::dtype(dtype).device(device));
  auto output = sampler(logits);
  EXPECT_TRUE(
      torch::allclose(output, logits.argmax(/*dim=*/-1, /*keepdim=*/true)));
}

TEST(SamplerTest, ToppTopk) {
  // Test GreedySampler
  torch::ScalarType dtype(torch::kFloat32);
  torch::Device device(torch::kCPU);
  SamplingParameters params;
  params.top_k = {10, 0};
  params.top_p = {0.9, 1.0};
  params.do_sample = {true, true};
  Sampler sampler(params, dtype, device);

  int64_t batch_size = 2;
  int64_t vocab_size = 30;
  const auto logits = torch::randn({batch_size, vocab_size},
                                   torch::dtype(dtype).device(device));
  auto output = sampler(logits);

  // TODO: add unit test for top_k and top_p
  // EXPECT_TRUE(
  //     torch::allclose(output, logits.argmax(/*dim=*/-1, /*keepdim=*/true)));
}

}  // namespace llm
