#include "rejection_sampler.h"

#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorImpl.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch/types.h>

#include <cstdint>

namespace llm {

TEST(RejectionSamplerTest, Greedy) {
  torch::ScalarType dtype(torch::kFloat32);
  torch::Device device(torch::kCPU);
  const auto options = torch::dtype(dtype).device(device);
  RejectionSampler sampler({false, false});

  int64_t batch_size = 2;
  int64_t n_speculative_tokens = 3;
  int64_t vocab_size = 4;
  int64_t n_bonus_tokens = 1;

  const auto draft_token_ids =
      torch::randint(0, vocab_size, {batch_size, n_speculative_tokens}, device);
  const auto draft_probs =
      torch::randn({batch_size, n_speculative_tokens, vocab_size}, options);
  const auto target_probs =
      torch::randn({batch_size, n_speculative_tokens, vocab_size}, options);
  const auto bonus_token_ids =
      torch::randint(0, vocab_size, {batch_size, n_bonus_tokens}, device);

  auto output =
      sampler(draft_token_ids, draft_probs, target_probs, bonus_token_ids);

  auto target_token_ids = torch::argmax(target_probs, /*dim=*/-1);
  auto accepted_token_ids =
      torch::cat({target_token_ids, bonus_token_ids}, /*dim=*/-1);
  EXPECT_TRUE(torch::allclose(output, accepted_token_ids));
}

TEST(RejectionSamplerTest, Random) {
  torch::ScalarType dtype(torch::kFloat32);
  torch::Device device(torch::kCPU);
  const auto options = torch::dtype(dtype).device(device);
  RejectionSampler sampler({true, true});

  int64_t batch_size = 2;
  int64_t n_speculative_tokens = 3;
  int64_t vocab_size = 4;
  int64_t n_bonus_tokens = 1;

  const auto draft_token_ids =
      torch::randint(0, vocab_size, {batch_size, n_speculative_tokens}, device);
  const auto draft_probs =
      torch::randn({batch_size, n_speculative_tokens, vocab_size}, options);
  const auto target_probs =
      torch::randn({batch_size, n_speculative_tokens, vocab_size}, options);
  const auto bonus_token_ids =
      torch::randint(0, vocab_size, {batch_size, n_bonus_tokens}, device);

  auto output =
      sampler(draft_token_ids, draft_probs, target_probs, bonus_token_ids);

  // TODO: add test for random rejection sampling
}

}  // namespace llm
