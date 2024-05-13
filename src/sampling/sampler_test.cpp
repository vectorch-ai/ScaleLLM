#include "sampler.h"

#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorImpl.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch/types.h>

namespace llm {

TEST(SamplerTest, Greedy) {
  // Test GreedySampler
  torch::ScalarType dtype(torch::kFloat32);
  torch::Device device(torch::kCPU);
  const auto options = torch::dtype(dtype).device(device);

  int64_t batch_size = 2;
  int64_t vocab_size = 32000;
  const auto probs =
      torch::randn({batch_size, vocab_size}, options).softmax(/*dim=*/-1);
  auto output = Sampler::greedy_sample(probs);
  const auto desired_output = probs.argmax(/*dim=*/-1);
  EXPECT_TRUE(torch::allclose(output, desired_output));
}

TEST(SamplerTest, Random) {
  // Test GreedySampler
  torch::ScalarType dtype(torch::kFloat32);
  torch::Device device(torch::kCPU);
  const auto options = torch::dtype(dtype).device(device);

  // set random seed
  torch::manual_seed(100);

  int64_t vocab_size = 50;
  int64_t num_samples = 500000;

  auto target_prob = torch::randn({vocab_size}, options).softmax(/*dim=*/-1);

  auto probs = target_prob.reshape({1, -1}).repeat({num_samples, 1});
  auto output = Sampler::random_sample(probs);

  auto token_ids = output.flatten();
  // calculate the probability of each sampled token
  auto bincount =
      token_ids.bincount(/*weights=*/torch::nullopt, /*minlength=*/vocab_size);
  auto sample_prob = bincount.to(torch::kFloat) / num_samples;

  EXPECT_TRUE(
      torch::allclose(target_prob, sample_prob, /*rtol=*/1e-2, /*atol=*/1e-3));
}

}  // namespace llm
