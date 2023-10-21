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

TEST(LogitsProcessorTest, TemperatureKernel) {
  // TODO: add test for TemperatureLogitsProcessorKernel
}

}  // namespace llm
