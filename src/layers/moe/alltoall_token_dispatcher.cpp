#include "alltoall_token_dispatcher.h"

#include "model_parallel/process_group.h"
#include "utils.h"

namespace llm {
namespace {

// returns [ep_size, n_experts]
torch::Tensor allgather(torch::Tensor tokens_per_expert,  // [n_experts]
                        ProcessGroup* pg) {
  if (pg == nullptr) {
    return tokens_per_expert;
  }

  auto size = pg->world_size();
  if (size == 1) {
    return tokens_per_expert;
  }

  // allocate output tensor: [ep_size, n_experts]
  auto output = torch::empty({size, tokens_per_expert.size(0)},
                             tokens_per_expert.options());
  pg->allgather(tokens_per_expert, output);
  return output;
}

torch::Tensor alltoall(const torch::Tensor& tokens,  // [n_tokens, dim]
                       const std::vector<int64_t>& input_splits,
                       const std::vector<int64_t>& output_splits,
                       ProcessGroup* pg) {
  if (pg == nullptr) {
    return tokens;
  }

  auto size = pg->world_size();
  if (size == 1) {
    return tokens;
  }

  // allocate output tensor based on output_splits
  int64_t output_size = std::reduce(output_splits.begin(), output_splits.end());
  int64_t dim = tokens.size(1);
  auto output = torch::empty({output_size, dim}, tokens.options());

  pg->alltoall(tokens, output, input_splits, output_splits);
  return output;
}

torch::Tensor sort_by_idxs(const torch::Tensor& tokens,  // [n_tokens, dim]
                           const std::vector<int64_t>& split_sizes,
                           const std::vector<int64_t>& sorted_idxs) {
  CHECK_EQ(split_sizes.size(), sorted_idxs.size())
      << "split_sizes and sorted_idxs should have the same size";

  auto chunks = tokens.split(split_sizes, /*dim=*/0);
  std::vector<torch::Tensor> sorted_chunks;
  sorted_chunks.reserve(sorted_idxs.size());
  for (const auto& idx : sorted_idxs) {
    sorted_chunks.push_back(chunks[idx]);
  }
  auto sorted_tokens = torch::cat(sorted_chunks, /*dim=*/0);
  return sorted_tokens;
}

std::vector<int64_t> to_vector(const torch::Tensor& tensor) {
  auto t = tensor.cpu();
  const int64_t* data_ptr = t.const_data_ptr<int64_t>();
  return {data_ptr, data_ptr + t.numel()};
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
      to_vector(input_chunk_idxs.reshape({-1, n_local_experts}).t().ravel());

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
      to_vector(input_chunk_idxs.reshape({n_local_experts, -1}).t().ravel());
}

// returns number of tokens for each local expert: [n_local_experts]
torch::Tensor AlltoAllTokenDispatcher::preprocess(
    const torch::Tensor& routing_map  // [n_tokens, n_experts]
) {
  // [n_tokens, n_experts] => [n_experts]
  auto local_tokens_per_expert = routing_map.sum(/*dim=*/0);
  if (ep_size_ <= 1) {
    return local_tokens_per_expert;
  }

  // calculate input_splits/output_splits for alltoall communication
  // [n_experts] => [ep_size, n_local_experts] => [ep_size]
  // input_splits: num of tokens to each rank
  this->input_splits_ =
      to_vector(local_tokens_per_expert.reshape({ep_size_, n_local_experts_})
                    .sum(/*dim=*/1));

  // gather the global distribution of tokens accross ranks
  auto tokens_per_expert = allgather(local_tokens_per_expert, ep_pg_);

  // slice tokens for local experts
  // [ep_size, n_experts] => [ep_size, n_local_experts]
  auto tokens_per_local_expert =
      tokens_per_expert.slice(/*dim=*/1,
                              /*start=*/n_local_experts_ * ep_rank_,
                              /*end=*/n_local_experts_ * (ep_rank_ + 1));
  // number of tokens from each rank
  // [ep_size, n_local_experts] => [ep_size]
  this->output_splits_ = to_vector(tokens_per_local_expert.sum(/*dim=*/1));

  this->tokens_per_local_expert_ = to_vector(tokens_per_local_expert.ravel());
  this->restore_tokens_per_local_expert_ =
      to_vector(tokens_per_local_expert.t().ravel());

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

  // [n_tokens, n_experts] => [n_local_experts]
  auto tokens_per_local_expert = preprocess(routing_map);

  // [n_tokens, n_experts] => [n_experts, n_tokens]
  this->routing_map_ = routing_map.t().contiguous();

  auto [local_permuted_tokens, sorted_indices] =
      permute(tokens, this->routing_map_);

  this->sorted_indices_ = sorted_indices;

  // [n_tokens, n_experts] => [n_experts, n_tokens] => [n_permuted_tokens]
  this->permuted_probs_ = probs.t().contiguous().masked_select(
      /*mask=*/this->routing_map_);

  // alltoall communication to gather tokens from all ranks
  // sorted by (ep_size, n_local_experts)
  auto permuted_tokens = alltoall(
      local_permuted_tokens, this->input_splits_, this->output_splits_, ep_pg_);

  // sort tokens by (n_local_experts, ep_size)
  if (n_local_experts_ > 1) {
    permuted_tokens = sort_by_idxs(permuted_tokens,
                                   this->tokens_per_local_expert_,
                                   this->sort_by_local_experts_);
  }
  return {permuted_tokens, tokens_per_local_expert};
}

torch::Tensor AlltoAllTokenDispatcher::combine(
    torch::Tensor permuted_tokens,     // [n_permuted_tokens, dim]
    std::optional<torch::Tensor> bias  // [n_tokens, n_active_experts]
) {
  // tokens in expert_output is already sorted by (n_local_experts, ep_size)
  if (n_local_experts_ > 1) {
    // sort by (ep_size, n_local_experts) for alltoall communication
    permuted_tokens = sort_by_idxs(permuted_tokens,
                                   this->restore_tokens_per_local_expert_,
                                   this->restore_output_by_local_experts_);
  }
  // alltoall communication to gather tokens from all ranks
  auto local_permuted_tokens = alltoall(permuted_tokens,
                                        /*input_splits=*/this->output_splits_,
                                        /*output_splits=*/this->input_splits_,
                                        ep_pg_);

  // apply weights for each expert
  // [n_permuted_tokens, dim] * [n_permuted_tokens]
  local_permuted_tokens *= this->permuted_probs_.unsqueeze(/*dim=*/-1);

  return unpermute(
      local_permuted_tokens, this->sorted_indices_, this->restore_shape_);
}

}  // namespace llm