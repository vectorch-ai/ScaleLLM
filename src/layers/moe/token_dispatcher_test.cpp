#include <c10/core/ScalarType.h>
#include <gtest/gtest.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/torch.h>

#include <cstdint>
#include <iostream>

#include "local_token_dispatcher.h"

namespace llm {

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
  auto routing_map = torch::zeros_like(logits)
                         .to(torch::kInt)
                         .scatter(
                             /*dim=*/1, /*index=*/indices, /*value=*/1)
                         .to(torch::kBool);
  auto probs = torch::zeros_like(logits).scatter(
      /*dim=*/1, /*index=*/indices, /*value=*/1.0 / n_topk);

  LocalTokenDispatcher dispatcher;
  auto [permuted_tokens, tokens_per_expert] =
      dispatcher.dispatch(tokens, probs, routing_map);

  // check shapes
  EXPECT_EQ(permuted_tokens.sizes(),
            torch::IntArrayRef({n_topk * n_topk, dim}));
  EXPECT_EQ(tokens_per_expert.sizes(), torch::IntArrayRef({n_experts}));

  auto expert_output = permuted_tokens;
  auto bias = std::nullopt;
  auto output = dispatcher.combine(expert_output, bias);

  EXPECT_TRUE(torch::allclose(output, tokens));
}

}  // namespace llm
