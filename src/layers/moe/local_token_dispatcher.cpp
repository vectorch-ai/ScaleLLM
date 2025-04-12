#include "local_token_dispatcher.h"

#include "permutation.h"

namespace llm {

// clang-format off
// for exmple: n_experts = 8, topk = 2
//  _____________________________________________________________________________
// |                         |  dispatch   |     Group-GEMM    |     combine     |
// |                         |_____________|___________________|_________________|
// |                         |  permute    |       GEMM        |    unpermute    |
// |_________________________|_____________|___________________|_________________|
// |        |                |             |                   |                 |
// |        | t1 -> [e0, e4] |  e0: t1     |  [t1]     -> e0   | [e0, e4] ->  t1 |
// |        | t2 -> [e4, e5] |  e1:        |  [t1, t2] -> e4   | [e4, e5] ->  t2 |
// |        |                |  e2:        |  [t2]     -> e5   |                 |
// |   d0   |                |  e3:        |                   |                 |
// |        |                |  e4: t1, t2 |                   |                 |
// |        |                |  e5: t2     |                   |                 |
// |        |                |  e6:        |                   |                 |
// |        |                |  e7:        |                   |                 |
// |________|________________|_____________|___________________|_________________|
// clang-format on

std::tuple<torch::Tensor, torch::Tensor> LocalTokenDispatcher::dispatch(
    torch::Tensor tokens,      // [n_tokens, dim]
    torch::Tensor probs,       // [n_tokens, n_experts]
    torch::Tensor routing_map  // [n_tokens, n_experts]
) {
  this->restore_shape_ = tokens.sizes();

  // [n_tokens, n_experts] => [n_experts]
  auto tokens_per_expert = routing_map.sum(/*dim=*/0);

  // [n_tokens, n_experts] => [n_experts, n_tokens]
  this->routing_map_ = routing_map.t().contiguous();

  auto [permuted_tokens, sorted_indices] = permute(tokens, this->routing_map_);

  this->sorted_indices_ = sorted_indices;

  // [n_tokens, n_experts] => [n_experts, n_tokens] => [n_permuted_tokens]
  this->permuted_probs_ = probs.t().contiguous().masked_select(
      /*mask=*/this->routing_map_);

  return {permuted_tokens, tokens_per_expert};
}

torch::Tensor LocalTokenDispatcher::combine(
    torch::Tensor permuted_tokens,     // [n_permuted_tokens, dim]
    std::optional<torch::Tensor> bias  // [n_tokens, n_active_experts]
) {
  // apply weights for each expert
  // [n_permuted_tokens, dim] * [n_permuted_tokens]
  permuted_tokens *= this->permuted_probs_.unsqueeze(/*dim=*/-1);

  return unpermute(
      permuted_tokens, this->sorted_indices_, this->restore_shape_);
}

}  // namespace llm