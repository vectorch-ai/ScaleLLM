#include "rejection_sampler.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch/types.h>

#include <cstdint>

#include "sampling/sampler.h"

namespace llm {

// TEST(RejectionSamplerTest, Basic) {
//   // test with hand-crafted example
//   torch::ScalarType dtype(torch::kFloat32);
//   torch::Device device(torch::kCPU);
//   const auto options = torch::dtype(dtype).device(device);

//   // set random seed
//   torch::manual_seed(100);

//   const auto draft_token_ids =
//       torch::tensor({{1, 2, 3}}, options.dtype(torch::kInt64));

//   // shape: [1, 3, 5]
//   auto draft_probs = torch::tensor({{0.2104, 0.2163, 0.1912, 0.1937, 0.1884},
//                                     {0.2100, 0.1803, 0.2398, 0.2088, 0.1610},
//                                     {0.1838, 0.2079, 0.2270, 0.2451,
//                                     0.1362}},
//                                    options)
//                          .reshape({1, 3, 5});
//   // shape: [1, 3, 5]
//   auto target_probs = torch::tensor({{0.1299, 0.2462, 0.1821, 0.1354,
//   0.3064},
//                                      {0.1159, 0.2839, 0.1603, 0.2451,
//                                      0.1949}, {0.0002, 0.0433, 0.6629,
//                                      0.1469, 0.1467}},
//                                     options)
//                           .reshape({1, 3, 5});

//   // selected_target_probs:  [0.2462  0.1603  0.1469]
//   // selected_draft_probs:   [0.2163  0.2398  0.2451]
//   // acceptance_probs:       [1.1382  0.6685  0.5993]
//   // uniform_rand:           [0.4785  0.6589  0.9399]
//   // accepted:               [  1        1       0  ]
//   auto uniform_rand = torch::tensor({{0.4785, 0.6589, 0.9399}}, options);
//   auto bonus_token_ids = torch::tensor({{5}}, options.dtype(torch::kInt64));

//   auto [output, masked_output] =
//       RejectionSampler::random_sample(draft_token_ids,
//                                       draft_probs,
//                                       target_probs,
//                                       uniform_rand,
//                                       bonus_token_ids,
//                                       true);
//   auto desired_output =
//       torch::tensor({{1, 2, 2, 5}}, options.dtype(torch::kInt64));
//   EXPECT_TRUE(torch::allclose(output, desired_output));

//   auto desired_masked_output =
//       torch::tensor({{1, 2, 2, -1}}, options.dtype(torch::kInt64));
//   EXPECT_TRUE(torch::allclose(masked_output, desired_masked_output));
// }

// TEST(RejectionSamplerTest, Mask) {
//   // test accepted mask
//   torch::ScalarType dtype(torch::kBool);
//   torch::Device device(torch::kCPU);
//   const auto options = torch::dtype(dtype).device(device);

//   // clang-format off
//   auto accepted = torch::tensor({
//         {0, 1, 0, 1},
//         {1, 0, 1, 1},
//         {1, 1, 0, 1},
//         {1, 1, 1, 1}},
//         options);
//   auto desired_mask = torch::tensor({
//         {1, 0, 0, 0, 0},
//         {1, 1, 0, 0, 0},
//         {1, 1, 1, 0, 0},
//         {1, 1, 1, 1, 1}},
//         options);
//   // clang-format on
//   auto mask = RejectionSampler::build_accepted_mask(accepted);
//   EXPECT_TRUE(torch::allclose(mask, desired_mask));
// }

// TEST(RejectionSamplerTest, Greedy) {
//   torch::ScalarType dtype(torch::kFloat32);
//   torch::Device device(torch::kCPU);
//   const auto options = torch::dtype(dtype).device(device);
//   const auto do_sample = torch::tensor({false, false}, device);
//   RejectionSampler sampler(do_sample);

//   int64_t batch_size = 2;
//   int64_t n_speculative_tokens = 3;
//   int64_t vocab_size = 4;
//   int64_t n_bonus_tokens = 1;

//   const auto draft_token_ids =
//       torch::randint(0,
//                      vocab_size,
//                      {batch_size, n_speculative_tokens},
//                      torch::dtype(torch::kInt64).device(device));
//   auto target_probs =
//       torch::randn({batch_size, n_speculative_tokens, vocab_size}, options)
//           .softmax(/*dim=*/-1, /*dtype=*/torch::kFloat32);
//   const auto bonus_token_ids =
//       torch::randint(0,
//                      vocab_size,
//                      {batch_size, n_bonus_tokens},
//                      torch::dtype(torch::kInt64).device(device));

//   auto [output, masked_output] =
//       RejectionSampler::greedy_sample(draft_token_ids,
//                                       target_probs,
//                                       bonus_token_ids,
//                                       /*mask_out_rejected_tokens=*/false);
//   EXPECT_FALSE(masked_output.defined());

//   const auto desired_output = target_probs.argmax(/*dim=*/-1);

//   // check target tokens
//   EXPECT_TRUE(torch::allclose(
//       output.slice(/*dim=*/-1, /*start=*/0, /*end=*/n_speculative_tokens),
//       desired_output));
//   // check bonus tokens
//   EXPECT_TRUE(
//       torch::allclose(output.slice(/*dim=*/-1,
//       /*start=*/n_speculative_tokens),
//                       bonus_token_ids));
// }

// TEST(RejectionSamplerTest, Random) {
//   torch::ScalarType dtype(torch::kFloat32);
//   torch::Device device(torch::kCPU);
//   const auto options = torch::dtype(dtype).device(device);

//   // set random seed
//   torch::manual_seed(100);

//   int64_t vocab_size = 50;
//   int64_t num_samples = 500000;

//   auto target_prob = torch::randn({vocab_size}, options).softmax(/*dim=*/-1);
//   auto target_probs =
//       target_prob.reshape({1, 1, -1}).repeat({num_samples, 1, 1});

//   auto draft_probs =
//       torch::randn({num_samples, 1, vocab_size},
//       options).softmax(/*dim=*/-1);
//   auto draft_token_ids = Sampler::random_sample(draft_probs);

//   // not used
//   auto bonus_token_ids =
//       torch::ones({num_samples, 1}, options.dtype(torch::kInt64));

//   auto uniform_rand = torch::rand(draft_token_ids.sizes(), options);
//   auto [output, masked_output] =
//       RejectionSampler::random_sample(draft_token_ids,
//                                       draft_probs,
//                                       target_probs,
//                                       uniform_rand,
//                                       bonus_token_ids,
//                                       false);
//   EXPECT_FALSE(masked_output.defined());

//   // remove bonus token
//   auto token_ids = output.slice(/*dim=*/-1, /*start=*/0,
//   /*end=*/-1).flatten();

//   // calculate the probability of each sampled token
//   auto bincount =
//       token_ids.bincount(/*weights=*/torch::nullopt,
//       /*minlength=*/vocab_size);
//   auto sample_prob = bincount.to(torch::kFloat) / num_samples;

//   EXPECT_TRUE(
//       torch::allclose(target_prob, sample_prob, /*rtol=*/1e-2,
//       /*atol=*/1e-3));
// }

TEST(RejectionSamplerTest, LogProbs) {
  torch::ScalarType dtype(torch::kFloat32);
  torch::Device device(torch::kCPU);
  const auto options = torch::dtype(dtype).device(device);
  const auto do_sample = torch::tensor({false, false}, device);
  RejectionSampler sampler(do_sample);

  int64_t batch_size = 2;
  int64_t n_speculative_tokens = 4;
  int64_t vocab_size = 8;

  const auto draft_token_ids =
      torch::randint(0,
                     vocab_size,
                     {batch_size, n_speculative_tokens},
                     torch::dtype(torch::kInt64).device(device));
  auto draft_probs =
      torch::randn({batch_size, n_speculative_tokens, vocab_size}, options)
          .softmax(/*dim=*/-1);

  auto target_logits =
      torch::randn({batch_size, n_speculative_tokens + 1, vocab_size}, options);
  const auto bonus_token_ids =
      torch::randint(0,
                     vocab_size,
                     {batch_size, 1},
                     torch::dtype(torch::kInt64).device(device));

  auto output = sampler.forward(
      draft_token_ids, draft_probs, target_logits, bonus_token_ids);

  EXPECT_TRUE(false);

  //   auto output = RejectionSampler::greedy_sample(target_probs);
  //   const auto desired_output = target_probs.argmax(/*dim=*/-1);

  //   // check target tokens
  //   EXPECT_TRUE(torch::equal(output, desired_output));
}

}  // namespace llm
