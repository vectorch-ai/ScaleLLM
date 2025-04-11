#include "local_token_dispatcher.h"

#include "utils.h"

namespace llm {

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