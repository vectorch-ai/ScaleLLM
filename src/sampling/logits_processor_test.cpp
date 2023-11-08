#include "logits_processor.h"

#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <gtest/gtest.h>
#include <kernels/sampling/sampling_kernels.h>
#include <torch/torch.h>

namespace llm {
TEST(LogitsProcessorTest, Temperature) {
  // Test TemperatureLogitsProcessor
  torch::ScalarType dtype(torch::kHalf);
  torch::Device device(torch::kCUDA);
  std::vector<float> temperatures = {0.5, 1.5};
  TemperatureLogitsProcessor processor(temperatures, dtype, device);

  int64_t batch_size = 2;
  int64_t vocab_size = 5;
  auto logits = torch::randn({batch_size, vocab_size},
                             torch::dtype(dtype).device(device));
  auto token_ids = torch::randint(/*high=*/vocab_size,
                                  {batch_size},
                                  torch::dtype(torch::kLong).device(device));
  auto logits_output = processor(token_ids, logits.clone());
  for (int i = 0; i < batch_size; ++i) {
    EXPECT_TRUE(torch::allclose(logits_output[i],
                                logits[i] / temperatures[i],
                                /*rtol=*/1e-3,
                                /*atol=*/1e-5));
  }
}

TEST(LogitsProcessorTest, TopK) {
  // Test TopKLogitsProcessor
  torch::ScalarType dtype(torch::kHalf);
  torch::Device device(torch::kCUDA);
  int64_t batch_size = 5;
  int64_t vocab_size = 1024;
  // random generate top_k
  std::vector<int64_t> top_k;
  for (int64_t i = 0; i < batch_size; ++i) {
    top_k.push_back(std::rand() % vocab_size);
  }
  const float filter_value = -std::numeric_limits<float>::infinity();
  TopKLogitsProcessor processor(top_k, dtype, device, filter_value);

  auto logits = torch::randn({batch_size, vocab_size},
                             torch::dtype(dtype).device(device));
  torch::Tensor token_ids;
  auto logits_output = processor(token_ids, logits.clone());

  for (int64_t i = 0; i < batch_size; ++i) {
    const int64_t k = top_k[i];

    // calculate top k values one by one
    auto [top_k_values, top_k_indices] =
        logits[i].topk(k, /*dim=*/-1, /*largest=*/true, /*sorted=*/true);
    auto [output_sorted, output_indices] =
        logits_output[i].sort(/*dim=*/-1, /*descending=*/true);
    // top k values should be the same
    ASSERT_TRUE(
        torch::equal(output_sorted.slice(/*dim=*/0, 0, k), top_k_values));
    // all remaining values should be filter_value
    auto masked_output = output_sorted.slice(/*dim=*/0, k);
    ASSERT_TRUE(torch::equal(masked_output,
                             torch::full_like(masked_output, filter_value)));
  }
}

TEST(LogitsProcessorTest, TopP) {
  // Test TopPLogitsProcessor
  torch::ScalarType dtype(torch::kHalf);
  torch::Device device(torch::kCUDA);
  int64_t batch_size = 4;
  int64_t vocab_size = 100;
  int64_t min_tokens_to_keep = 1;
  std::vector<float> top_p = {0.5, 0.7, 0.9, 1.0};
  const float filter_value = -std::numeric_limits<float>::infinity();
  TopPLogitsProcessor processor(
      top_p, dtype, device, filter_value, min_tokens_to_keep);

  auto logits = torch::randn({batch_size, vocab_size},
                             torch::dtype(dtype).device(device));
  torch::Tensor token_ids;
  auto logits_output = processor(token_ids, logits.clone());

  // verify result one by one
  for (int64_t i = 0; i < batch_size; ++i) {
    const float p = top_p[i];

    // calculate number of values to keep (k)
    auto probs = torch::softmax(logits[i], /*dim=*/-1);
    // calculate top p values one by one
    auto [sorted_probs, sorted_indices] =
        probs.sort(/*dim=*/-1, /*descending=*/true);
    auto probs_sum = sorted_probs.cumsum(/*dim=*/-1);
    torch::Tensor mask = (probs_sum - sorted_probs) <= p;
    const int64_t k = std::max(mask.sum().item<int64_t>(), min_tokens_to_keep);

    // gather top logits value
    auto [sorted_logits, sorted_logits_indices] =
        logits[i].sort(/*dim=*/-1, /*descending=*/true);

    auto [output_sorted, output_indices] =
        logits_output[i].sort(/*dim=*/-1, /*descending=*/true);
    ASSERT_TRUE(torch::equal(output_sorted.slice(/*dim=*/0, 0, k),
                             sorted_logits.slice(/*dim=*/0, 0, k)));
    // all remaining values should be filter_value
    auto masked_output = output_sorted.slice(/*dim=*/0, k);
    ASSERT_TRUE(torch::equal(masked_output,
                             torch::full_like(masked_output, filter_value)));
  }
}

}  // namespace llm
