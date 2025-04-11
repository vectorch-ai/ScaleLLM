#include "permutation.h"

namespace llm {

std::tuple<torch::Tensor, torch::Tensor> permute(
    const torch::Tensor& tokens,      // [n_tokens, dim]
    const torch::Tensor& routing_map  // [n_experts, n_tokens]
) {
  auto n_tokens = routing_map.size(1);
  auto n_experts = routing_map.size(0);

  // [n_experts, n_tokens]
  auto token_indices =
      torch::arange(n_tokens, tokens.options().dtype(torch::kLong))
          .unsqueeze(/*dim=*/0)
          .expand({n_experts, n_tokens});

  // original token incides, sorted by expert idx
  auto sorted_indices = token_indices.masked_select(/*mask=*/routing_map);

  auto permuted_tokens = tokens.index_select(/*dim=*/0, sorted_indices);
  return {permuted_tokens, sorted_indices};
}

torch::Tensor unpermute(
    const torch::Tensor& permuted_tokens,    // [n_permuted_tokens, dim]
    const torch::Tensor& sorted_indices,     // [n_permuted_tokens]
    const torch::IntArrayRef& restore_shape  // [n_tokens, dim]
) {
  const auto dim = permuted_tokens.size(1);
  // [n_tokens, dim]
  auto output = torch::zeros(restore_shape, permuted_tokens.options());

  // [n_permuted_tokens] => [n_permuted_tokens, dim]
  auto index = sorted_indices.unsqueeze(/*dim=*/1).expand({-1, dim});

  // [n_permuted_tokens, dim] => [n_tokens, dim]
  output.scatter_add_(/*dim=*/0, /*index=*/index, /*src=*/permuted_tokens);
  return output;
}

}  // namespace llm