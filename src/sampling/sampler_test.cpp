#include "sampler.h"

#include <ATen/ops/equal.h>
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
  EXPECT_EQ(output.sizes(), torch::IntArrayRef({batch_size}));

  const auto desired_output = probs.argmax(/*dim=*/-1);
  EXPECT_TRUE(torch::allclose(output, desired_output));
}

TEST(SamplerTest, Logprobs) {
  torch::ScalarType dtype(torch::kFloat32);
  torch::Device device(torch::kCPU);
  const auto options = torch::dtype(dtype).device(device);
  const int64_t batch_size = 8;
  const int64_t vocab_size = 64000;
  const auto do_sample = torch::tensor(
      {true, false, true, true, false, false, true, false}, device);

  const int64_t top_logprobs = 20;
  Sampler sampler(do_sample, /*logprobs=*/true, top_logprobs);

  const auto logits = torch::randn({batch_size, vocab_size}, options);

  auto output = sampler(logits);

  EXPECT_EQ(output.next_tokens.sizes(), torch::IntArrayRef({batch_size}));
  EXPECT_EQ(output.logprobs.sizes(), torch::IntArrayRef({batch_size}));
  EXPECT_EQ(output.top_logprobs.sizes(),
            torch::IntArrayRef({batch_size, top_logprobs}));
  EXPECT_EQ(output.top_tokens.sizes(),
            torch::IntArrayRef({batch_size, top_logprobs}));

  const auto logprobs =
      torch::log_softmax(logits, /*dim=*/-1, /*dtype=*/torch::kFloat32);
  auto selected_tokens = output.next_tokens;
  auto selected_logprobs =
      logprobs.gather(/*dim=*/-1, selected_tokens.view({-1, 1}));
  EXPECT_TRUE(torch::equal(output.logprobs, selected_logprobs.view({-1})));

  auto [top_k_values, top_k_indices] = logprobs.topk(
      top_logprobs, /*dim=*/-1, /*largest=*/true, /*sorted=*/true);
  EXPECT_TRUE(torch::equal(output.top_logprobs, top_k_values));
  EXPECT_TRUE(torch::equal(output.top_tokens, top_k_indices));
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
  EXPECT_EQ(output.sizes(), torch::IntArrayRef({num_samples}));

  auto token_ids = output.flatten();
  // calculate the probability of each sampled token
  auto bincount = token_ids.bincount(/*weights=*/torch::nullopt,
                                     /*minlength=*/vocab_size);
  auto sample_prob = bincount.to(torch::kFloat) / num_samples;

  EXPECT_TRUE(torch::allclose(target_prob,
                              sample_prob,
                              /*rtol=*/1e-2,
                              /*atol=*/1e-3));
}

}  // namespace llm
