#pragma once

#include <torch/torch.h>

// Inspired by Megatron-LM: https://github.com/NVIDIA/Megatron-LM

namespace llm {
//
// TokenDispatcher is an abstract class that responsible for dispatching tokens
// to experts based on the routing map and combining expert outputs back to the
// original token order. It is used in the mixture of experts (MoE) layer.
//
// Example usage in MoE:
//  auto [probs, routing_map] = router(tokens);
//  auto [permuted_tokens, tokens_per_expert] = dispatcher->dispatch(
//      tokens, probs, routing_map);
//  auto [expert_output, bias] = experts(permuted_tokens, tokens_per_expert);
//  auto output = dispatcher->combine(expert_output, bias);
//  output += shared_experts(tokens);
//  return output;
//
// TODO: implement concrete classes for different dispatching strategies.
// 1> local token dispatcher
// 2> all-2-all token dispatcher for expert parallelism + tensor parallelism?
//
class TokenDispatcher {
 public:
  virtual ~TokenDispatcher() = default;

  // Dispatches tokens to experts based on the routing map.
  // Retruns a tuple of permuted_tokens and the tokens_per_expert.
  //  * permuted_tokens: [n_permuted_tokens, dim] sorted by expert
  //  * tokens_per_expert: [n_experts]
  virtual std::tuple<torch::Tensor, torch::Tensor> dispatch(
      torch::Tensor tokens,      // [n_tokens, dim]
      torch::Tensor probs,       // [n_tokens, n_experts]
      torch::Tensor routing_map  // [n_tokens, n_experts]
      ) = 0;

  // Combines expert outputs based on the routing map, applying the bias if
  // provided.
  // Returns the unpermuted activations.
  //  * unpermuted_output: [n_tokens, dim]
  virtual torch::Tensor combine(
      torch::Tensor expert_output,       // [n_permuted_tokens, dim]
      std::optional<torch::Tensor> bias  // [n_tokens, n_active_experts]
      ) = 0;
};

}  // namespace llm