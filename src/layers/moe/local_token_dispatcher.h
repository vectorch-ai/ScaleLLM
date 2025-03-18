#include <torch/torch.h>

#include "token_dispatcher.h"

namespace llm {

class LocalTokenDispatcher : public TokenDispatcher {
 public:
  LocalTokenDispatcher() = default;

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
  torch::Tensor sorted_indices_;
  torch::Tensor routing_map_;
  torch::IntArrayRef restore_shape_;
};

}  // namespace llm