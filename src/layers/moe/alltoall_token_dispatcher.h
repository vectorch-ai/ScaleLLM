#include <torch/torch.h>

#include <cstdint>

#include "model_parallel/process_group.h"
#include "token_dispatcher.h"

namespace llm {

// only support expert parallelism for now
class AlltoAllTokenDispatcher : public TokenDispatcher {
 public:
  // Constructors
  AlltoAllTokenDispatcher() = default;

  AlltoAllTokenDispatcher(int64_t n_experts,
                          int64_t n_local_experts,
                          ProcessGroup* ep_pg);

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
  // calculate input and output splits for alltoall communication
  // returns number of tokens for each local expert: [n_local_experts]
  torch::Tensor preprocess(
      const torch::Tensor& routing_map  // [n_tokens, n_experts];
  );

  int64_t ep_size_;
  int64_t ep_rank_;
  int64_t n_experts_;
  int64_t n_local_experts_;

  ProcessGroup* ep_pg_;

  // original token incides, sorted by expert idx
  // [n_permuted_tokens]
  torch::Tensor sorted_indices_;
  // [n_experts, n_tokens]
  torch::Tensor routing_map_;
  // the original shape for unpermutation
  // [n_tokens, dim]
  torch::IntArrayRef restore_shape_;
  // [n_permuted_tokens]
  torch::Tensor permuted_probs_;

  // num of tokens to each rank
  // [ep_size]
  torch::Tensor input_splits_;
  // num of tokens from each rank
  // [ep_size]
  torch::Tensor output_splits_;

  // num of tokens from each rank for local experts
  // [ep_size, n_local_experts]
  torch::Tensor tokens_per_local_expert_;

  // [n_local_experts*ep_size]
  torch::Tensor sort_by_local_experts_;
  // [ep_size*n_local_experts]
  torch::Tensor restore_output_by_local_experts_;
};

}  // namespace llm