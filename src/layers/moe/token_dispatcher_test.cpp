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

TEST(TokenDispatcherTest, Local) {
  const auto dtype = torch::kFloat;
  const auto device = torch::kCUDA;
  const auto options = torch::dtype(dtype).device(device);

  const int64_t dim = 8;
  const int64_t n_tokens = 2;
  const int64_t n_experts = 4;
  const int64_t n_topk = 2;

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

  EXPECT_TRUE(torch::allclose(output, tokens));
}

TEST(TokenDispatcherTest, AlltoAll) {
  const int64_t world_size = 1;
  const auto device_type = torch::kCUDA;

  run_collective_test(world_size, device_type, [](const ProcessGroup* pg) {
    const auto dtype = torch::kFloat;
    const auto device = torch::kCUDA;

    const auto options = torch::dtype(dtype).device(device);

    const int64_t dim = 8;
    const int64_t n_tokens = 2;
    const int64_t n_experts = 4;
    const int64_t n_topk = 2;

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

    // check shapes
    EXPECT_EQ(permuted_tokens.sizes(),
              torch::IntArrayRef({n_tokens * n_topk, dim}));
    EXPECT_EQ(tokens_per_expert.sizes(), torch::IntArrayRef({n_experts}));

    auto bias = std::nullopt;
    auto output = dispatcher.combine(permuted_tokens, bias);

    EXPECT_TRUE(torch::allclose(output, tokens));
  });
}

}  // namespace llm
