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
  Sampler sampler({false, false});

  int64_t batch_size = 2;
  int64_t vocab_size = 32000;
  const auto logits = torch::randn({batch_size, vocab_size}, options);
  auto output = sampler(logits);

  const auto probs =
      torch::softmax(logits, /*dim=*/-1, /*dtype=*/torch::kFloat32);
  EXPECT_TRUE(torch::allclose(output.next_tokens, probs.argmax(/*dim=*/-1)));
}

}  // namespace llm
