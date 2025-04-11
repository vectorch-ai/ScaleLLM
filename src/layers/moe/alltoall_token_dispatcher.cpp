#include "alltoall_token_dispatcher.h"

namespace llm {

std::tuple<torch::Tensor, torch::Tensor> AlltoAllTokenDispatcher::dispatch(
    torch::Tensor tokens,      // [n_tokens, dim]
    torch::Tensor probs,       // [n_tokens, n_experts]
    torch::Tensor routing_map  // [n_tokens, n_experts]
) {
  this->restore_shape_ = tokens.sizes();
  auto n_tokens = tokens.size(0);
  auto n_experts = probs.size(1);

  // [n_tokens, n_experts] => [n_experts]
  auto tokens_per_expert = routing_map.sum(/*dim=*/0);

  // [n_tokens, n_experts] => [n_experts, n_tokens]
  this->routing_map_ = routing_map.t().contiguous();

  // [n_experts, n_tokens]
  auto token_indices =
      torch::arange(n_tokens, tokens.options().dtype(torch::kLong))
          .unsqueeze(/*dim=*/0)
          .expand({n_experts, n_tokens});

  // original token incides, sorted by expert idx
  this->sorted_indices_ =
      token_indices.masked_select(/*mask=*/this->routing_map_);

  // [n_tokens, n_experts] => [n_experts, n_tokens] => [n_permuted_tokens]
  this->permuted_probs_ = probs.t().contiguous().masked_select(
      /*mask=*/this->routing_map_);

  auto permuted_tokens = tokens.index_select(/*dim=*/0, this->sorted_indices_);

  return {permuted_tokens, tokens_per_expert};
}

torch::Tensor AlltoAllTokenDispatcher::combine(
    torch::Tensor expert_output,       // [n_permuted_tokens, dim]
    std::optional<torch::Tensor> bias  // [n_tokens, n_active_experts]
) {
  const auto dim = expert_output.size(1);

  // apply weights for each expert
  // [n_permuted_tokens, dim] * [n_permuted_tokens]
  expert_output = expert_output * this->permuted_probs_.unsqueeze(/*dim=*/-1);

  // [n_tokens, dim]
  auto output = torch::zeros(this->restore_shape_, expert_output.options());

  // [n_permuted_tokens] => [n_permuted_tokens, dim]
  auto index = this->sorted_indices_.unsqueeze(/*dim=*/1).expand({-1, dim});

  // [n_permuted_tokens, dim] => [n_tokens, dim]
  output.scatter_add_(/*dim=*/0, index, expert_output);

  return output;
}

}  // namespace llm