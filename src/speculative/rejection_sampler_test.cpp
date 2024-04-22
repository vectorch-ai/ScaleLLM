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
  const auto do_sample = torch::tensor({false, false}, device);
  RejectionSampler sampler(do_sample);

  int64_t batch_size = 2;
  int64_t n_speculative_tokens = 3;
  int64_t vocab_size = 4;
  int64_t n_bonus_tokens = 1;

  const auto draft_token_ids =
      torch::randint(0,
                     vocab_size,
                     {batch_size, n_speculative_tokens},
                     torch::dtype(torch::kInt64).device(device));
  auto draft_probs =
      torch::randn({batch_size, n_speculative_tokens, vocab_size}, options)
          .softmax(/*dim=*/-1, /*dtype=*/torch::kFloat32);
  auto target_probs =
      torch::randn({batch_size, n_speculative_tokens, vocab_size}, options)
          .softmax(/*dim=*/-1, /*dtype=*/torch::kFloat32);
  const auto bonus_token_ids =
      torch::randint(0,
                     vocab_size,
                     {batch_size, n_bonus_tokens},
                     torch::dtype(torch::kInt64).device(device));

  auto output =
      sampler(draft_token_ids, draft_probs, target_probs, bonus_token_ids);

  // TODO: add test for greedy rejection sampling
}

TEST(RejectionSamplerTest, Random) {
  torch::ScalarType dtype(torch::kFloat32);
  torch::Device device(torch::kCPU);
  const auto options = torch::dtype(dtype).device(device);
  const auto do_sample = torch::tensor({true, false}, device);
  RejectionSampler sampler(do_sample);

  int64_t batch_size = 2;
  int64_t n_speculative_tokens = 3;
  int64_t vocab_size = 20;
  int64_t n_bonus_tokens = 1;

  const auto draft_token_ids =
      torch::randint(0,
                     vocab_size,
                     {batch_size, n_speculative_tokens},
                     torch::dtype(torch::kInt64).device(device));
  auto draft_probs =
      torch::randn({batch_size, n_speculative_tokens, vocab_size}, options)
          .softmax(/*dim=*/-1, /*dtype=*/torch::kFloat32);
  auto target_probs =
      torch::randn({batch_size, n_speculative_tokens, vocab_size}, options)
          .softmax(/*dim=*/-1, /*dtype=*/torch::kFloat32);
  const auto bonus_token_ids =
      torch::randint(0,
                     vocab_size,
                     {batch_size, n_bonus_tokens},
                     torch::dtype(torch::kInt64).device(device));

  auto output =
      sampler(draft_token_ids, draft_probs, target_probs, bonus_token_ids);

  // TODO: add test for random rejection sampling
}

}  // namespace llm
