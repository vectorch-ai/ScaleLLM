#include <torch/torch.h>

#include "token_dispatcher.h"

namespace llm {

class AlltoAllTokenDispatcher : public TokenDispatcher {
 public:
 AlltoAllTokenDispatcher() = default;

  std::tuple<torch::Tensor, torch::Tensor> dispatch(
      torch::Tensor tokens,      // [n_tokens, dim]
      torch::Tensor probs,       // [n_tokens, n_experts]
      torch::Tensor routing_map  // [n_tokens, n_experts]
      ) override;

  torch::Tensor combine(
      torch::Tensor expert_output,       // [n_permuted_tokens, dim]
      std::optional<torch::Tensor> bias  // [n_tokens, n_active_experts]
      ) override;

 private:
  // [n_permuted_tokens]
  torch::Tensor sorted_indices_;
  // [n_experts, n_tokens]
  torch::Tensor routing_map_;
  // [n_tokens, dim]
  torch::IntArrayRef restore_shape_;
  // [n_permuted_tokens]
  torch::Tensor permuted_probs_;
};

}  // namespace llm