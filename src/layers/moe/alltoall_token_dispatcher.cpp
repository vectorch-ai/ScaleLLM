#include "alltoall_token_dispatcher.h"

#include "model_parallel/process_group.h"
#include "utils.h"

namespace llm {
namespace {

// returns [ep_size, n_experts]
torch::Tensor gather_tokens_per_expert(
    torch::Tensor tokens_per_expert,  // [n_experts]
    ProcessGroup* pg) {
  auto size = pg->world_size();
  if (size == 1) {
    return tokens_per_expert;
  }
  // Create a tensor to hold the gathered data: [ep_size, n_experts]
  auto output = torch::empty({size, tokens_per_expert.size(0)},
                             tokens_per_expert.options());
  pg->allgather(tokens_per_expert, output);
  return output;
}

}  // namespace

AlltoAllTokenDispatcher::AlltoAllTokenDispatcher(int64_t n_experts,
                                                 int64_t n_local_experts,
                                                 ProcessGroup* ep_pg)
    : n_experts_(n_experts), n_local_experts_(n_local_experts), ep_pg_(ep_pg) {
  CHECK_EQ(n_experts % n_local_experts, 0)
      << "n_experts should be divisible by n_local_experts";
  ep_size_ = ep_pg_->world_size();
  ep_rank_ = ep_pg_->rank();

  // [n_experts]: [0, 1, 2, 3, 4, 5, 6, 7, ...]
  auto input_chunk_idxs = torch::arange(n_experts);
  // [n_experts] => [n_local_experts*ep_size]
  // => [ep_size, n_local_experts]
  // => [n_local_experts, ep_size]
  // => [n_local_experts*ep_size]
  // for example: n_experts = 8, n_local_experts = 2, ep_size = 4
  // [0, 1, 2, 3, 4, 5, 6, 7]
  // => [[0, 1], [2, 3], [4, 5], [6, 7]]
  // => [[0, 2, 4, 6], [1, 3, 5, 7]]
  // => [0, 2, 4, 6, 1, 3, 5, 7]
  this->sort_by_local_experts_ =
      input_chunk_idxs.reshape({-1, n_local_experts}).t().ravel();

  // [n_experts] => [ep_size * n_local_experts]
  // => [n_local_experts, ep_size]
  // => [ep_size, n_local_experts]
  // => [ep_size*n_local_experts]
  // for example: n_experts = 8, n_local_experts = 2, ep_size = 4
  // [0, 1, 2, 3, 4, 5, 6, 7]
  // => [[0,1, 2, 3], [4, 5, 6, 7]]
  // => [[0, 4], [1, 5], [2, 6], [3, 7]]
  // => [0, 4, 1, 5, 2, 6, 3, 7]
  this->restore_output_by_local_experts_ =
      input_chunk_idxs.reshape({n_local_experts, -1}).t().ravel();
}

// returns number of tokens for each local expert: [n_local_experts]
torch::Tensor AlltoAllTokenDispatcher::preprocess(
    const torch::Tensor& routing_map  // [n_tokens, n_experts]
) {
  // [n_tokens, n_experts] => [n_experts]
  auto local_tokens_per_expert = routing_map.sum(/*dim=*/0);

  // calculate input_splits/output_splits for alltoall communication
  // [n_experts] => [ep_size, n_local_experts] => [ep_size]
  // input_splits: num of tokens to each rank
  this->input_splits_ =
      local_tokens_per_expert.reshape({ep_size_, n_local_experts_})
          .sum(/*dim=*/1);

  // gather the global distribution of tokens accross ranks
  auto tokens_per_expert =
      gather_tokens_per_expert(local_tokens_per_expert, ep_pg_);

  // slice tokens for local experts
  // [ep_size, n_experts] => [ep_size, n_local_experts]
  auto tokens_per_local_expert =
      tokens_per_expert.slice(/*dim=*/1,
                              /*start=*/n_local_experts_ * ep_rank_,
                              /*end=*/n_local_experts_ * (ep_rank_ + 1));
  this->tokens_per_local_expert_ = tokens_per_local_expert;

  // number of tokens from each rank
  // [ep_size, n_local_experts] => [ep_size]
  this->output_splits_ = tokens_per_local_expert.sum(/*dim=*/1);

  // number of tokens for each local expert
  // [ep_size, n_local_experts] => [n_local_experts]
  return tokens_per_local_expert.sum(/*dim=*/0);
}

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

  auto [permuted_tokens, sorted_indices] = permute(tokens, this->routing_map_);

  this->sorted_indices_ = sorted_indices;

  // [n_tokens, n_experts] => [n_experts, n_tokens] => [n_permuted_tokens]
  this->permuted_probs_ = probs.t().contiguous().masked_select(
      /*mask=*/this->routing_map_);

  return {permuted_tokens, tokens_per_expert};
}

torch::Tensor AlltoAllTokenDispatcher::combine(
    torch::Tensor expert_output,       // [n_permuted_tokens, dim]
    std::optional<torch::Tensor> bias  // [n_tokens, n_active_experts]
) {
  // apply weights for each expert
  // [n_permuted_tokens, dim] * [n_permuted_tokens]
  expert_output = expert_output * this->permuted_probs_.unsqueeze(/*dim=*/-1);

  return unpermute(expert_output, this->sorted_indices_, this->restore_shape_);
}

}  // namespace llm