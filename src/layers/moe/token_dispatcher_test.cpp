#include <absl/random/random.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>

#include "alltoall_token_dispatcher.h"
#include "local_token_dispatcher.h"
#include "model_parallel/process_group.h"

namespace llm {
namespace {
void run_collective_test(int world_size,
                         torch::DeviceType device_type,
                         std::function<void(const ProcessGroup* pg)> func) {
  // create process groups
  std::vector<torch::Device> devices;
  devices.reserve(world_size);
  for (int i = 0; i < world_size; ++i) {
    devices.emplace_back(device_type, i);
  }
  auto process_groups = ProcessGroup::create_process_groups(devices);
  EXPECT_EQ(process_groups.size(), world_size);

  // run collective test in parallel
  std::vector<std::thread> threads;
  threads.reserve(process_groups.size());
  for (int i = 0; i < world_size; ++i) {
    ProcessGroup* pg = process_groups[i].get();
    threads.emplace_back([func, pg]() { func(pg); });
  }

  // wait for all threads to finish
  for (auto& thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }
}
}  // namespace

class TokenDispatcherTest
    : public ::testing::TestWithParam<std::tuple<torch::DeviceType,
                                                 torch::ScalarType,
                                                 int64_t /*dim*/,
                                                 int64_t /*max_tokens*/,
                                                 int64_t /*n_experts*/,
                                                 int64_t /*n_topk*/
                                                 >> {};

TEST_P(TokenDispatcherTest, Local) {
  const auto& [device_type, dtype, dim, max_tokens, n_experts, n_topk] =
      GetParam();
  absl::BitGen gen;
  const int64_t n_tokens =
      absl::Uniform<int64_t>(absl::IntervalClosedClosed, gen, 1, max_tokens);

  const auto options = torch::dtype(dtype).device(device_type);

  auto tokens = torch::randn({n_tokens, dim}, options);
  auto logits = torch::rand({n_tokens, n_experts}, options);
  auto [weights, indices] = logits.topk(n_topk, /*dim=*/-1);

  weights = torch::softmax(weights, /*dim=*/-1);
  // construct dense routing map and probs
  auto probs = torch::zeros_like(logits).scatter(
      /*dim=*/1, /*index=*/indices, /*value=*/1.0 / n_topk);
  auto routing_map = torch::zeros_like(logits, torch::kInt)
                         .scatter(
                             /*dim=*/1, /*index=*/indices, /*value=*/1)
                         .to(torch::kBool);

  LocalTokenDispatcher dispatcher;
  auto [permuted_tokens, tokens_per_expert] =
      dispatcher.dispatch(tokens, probs, routing_map);

  // check shapes
  EXPECT_EQ(permuted_tokens.sizes(),
            torch::IntArrayRef({n_tokens * n_topk, dim}));
  EXPECT_EQ(tokens_per_expert.sizes(), torch::IntArrayRef({n_experts}));

  auto bias = std::nullopt;
  auto output = dispatcher.combine(permuted_tokens, bias);
  EXPECT_TRUE(torch::allclose(output, tokens, /*rtol=*/1e-5, /*atol=*/1e-7));
}

TEST_P(TokenDispatcherTest, AlltoAll) {
  const auto& [device_type, dtype, dim, max_tokens, n_experts, n_topk] =
      GetParam();
  absl::BitGen gen;

  for (int world_size = 1; world_size <= torch::cuda::device_count();
       world_size *= 2) {
    run_collective_test(world_size, device_type, [&](const ProcessGroup* pg) {
      const auto& device = pg->device();
      const auto options = torch::dtype(dtype).device(device);
      const int64_t n_tokens = absl::Uniform<int64_t>(
          absl::IntervalClosedClosed, gen, 1, max_tokens);

      auto tokens = torch::randn({n_tokens, dim}, options);
      auto logits = torch::rand({n_tokens, n_experts}, options);
      auto [weights, indices] = logits.topk(n_topk, /*dim=*/-1);

      weights = torch::softmax(weights, /*dim=*/-1);
      // construct dense routing map and probs
      auto probs = torch::zeros_like(logits).scatter(
          /*dim=*/1, /*index=*/indices, /*value=*/1.0 / n_topk);
      auto routing_map = torch::zeros_like(logits)
                             .to(torch::kInt)
                             .scatter(
                                 /*dim=*/1, /*index=*/indices, /*value=*/1)
                             .to(torch::kBool);

      AlltoAllTokenDispatcher dispatcher(n_experts, pg);

      auto [permuted_tokens, tokens_per_expert] =
          dispatcher.dispatch(tokens, probs, routing_map);

      auto bias = std::nullopt;
      auto output = dispatcher.combine(permuted_tokens, bias);
      EXPECT_TRUE(
          torch::allclose(output, tokens, /*rtol=*/1e-5, /*atol=*/1e-7));
    });
  }
}

INSTANTIATE_TEST_SUITE_P(
    MoE,
    TokenDispatcherTest,
    ::testing::Combine(::testing::Values(torch::kCUDA),
                       ::testing::Values(torch::kHalf, torch::kBFloat16),
                       ::testing::Values(32),     // dim
                       ::testing::Values(16),     // max_tokens
                       ::testing::Values(8, 64),  // n_experts
                       ::testing::Values(1, 4)    // n_topk
                       ));

}  // namespace llm
