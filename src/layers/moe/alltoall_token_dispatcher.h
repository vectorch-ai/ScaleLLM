#include <torch/torch.h>

#include <cstdint>

#include "token_dispatcher.h"

namespace llm {
// forward declaration
class ProcessGroup;

// only support expert parallelism for now
class AlltoAllTokenDispatcher : public TokenDispatcher {
 public:
  // Constructors
  AlltoAllTokenDispatcher(int64_t n_experts,
                          const ProcessGroup* ep_pg);

  std::tuple<torch::Tensor, torch::Tensor> dispatch(
      torch::Tensor tokens,      // [n_tokens, dim]
      torch::Tensor probs,       // [n_tokens, n_experts] float tensor
      torch::Tensor routing_map  // [n_tokens, n_experts] bool tensor
      ) override;

  torch::Tensor combine(
      torch::Tensor permuted_tokens,  // [n_permuted_tokens, dim]
      std::optional<torch::Tensor>
          bias  // [n_tokens, n_active_experts] float tensor
      ) override;

 private:
  // calculate input and output splits for alltoall communication
  // returns number of tokens for each local expert: [n_local_experts]
  torch::Tensor preprocess(
      const torch::Tensor& routing_map  // [n_tokens, n_experts]
  );

  int64_t n_local_experts_ = 0;
  const ProcessGroup* ep_pg_ = nullptr;

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

  // metadata for alltoall communication
  // num of tokens to each rank
  // [ep_size]
  std::vector<int64_t> input_split_sizes_;
  // num of tokens from each rank
  // [ep_size]
  std::vector<int64_t> output_split_sizes_;

  // metadata for token sorting
  // num of tokens from each rank for local experts
  // sorted by [ep_size*n_local_experts]
  std::vector<int64_t> tokens_per_local_expert_;
  // sorted by [n_local_experts*ep_size]
  std::vector<int64_t> restore_tokens_per_local_expert_;

  // [n_local_experts*ep_size]
  std::vector<int64_t> sort_by_local_experts_;
  // [ep_size*n_local_experts]
  std::vector<int64_t> restore_output_by_local_experts_;
};

}  // namespace llm