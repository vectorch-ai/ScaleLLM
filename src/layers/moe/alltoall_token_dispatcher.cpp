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
  std::vector<int64_t> sizes = tokens_per_expert.sizes().vec();
  sizes[0] *= size;
  auto output = torch::empty(sizes, tokens_per_expert.options());
  // allgather tokens_per_expert
  // [n_experts] => [ep_size, n_experts]
  pg->allgather(tokens_per_expert, output);
  return output;
}

torch::Tensor alltoall(const torch::Tensor& tokens,  // [n_tokens, dim]
                       const std::vector<int64_t>& input_split_sizes,
                       const std::vector<int64_t>& output_split_sizes,
                       ProcessGroup* pg) {
  if (pg == nullptr) {
    return tokens;
  }

  auto size = pg->world_size();
  if (size == 1) {
    return tokens;
  }

  // allocate output tensor based on output_splits
  int64_t output_size =
      std::reduce(output_split_sizes.begin(), output_split_sizes.end());
  std::vector<int64_t> sizes = tokens.sizes().vec();
  sizes[0] = output_size;
  auto output = torch::empty(sizes, tokens.options());

  pg->alltoall(tokens, output, input_split_sizes, output_split_sizes);
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
  return torch::cat(sorted_chunks, /*dim=*/0);
}

std::vector<int64_t> to_vector(const torch::Tensor& tensor) {
  // ensure tensor is on CPU
  auto t = tensor.cpu();
  const int64_t* data_ptr = t.const_data_ptr<int64_t>();
  return {data_ptr, data_ptr + t.numel()};
}

}  // namespace

// clang-format off
// for exmple: n_experts = 8, n_local_experts = 2, ep_size = 4, topk = 2
//  _______________________________________________________________________________________________________________________________________
// |                         |               dispatch                    |     Group-GEMM    |                 combine                     |
// |                         |___________________________________________|___________________|_____________________________________________|
// |                         |  permute  |   all2all     |  sort_by_idx  |       GEMM        | sort_by_idx   |  all2all  |    unpermute    |
// |_________________________|___________|_______________|_______________|___________________|_______________|___________|_________________|
// |        |  sorted by     | (experts) |(rank, experts)|(experts, rank)|                   |(rank, experts)| (experts) |                 |
// |________|________________|___________|_______________|_______________|___________________|_______________|___________|_________________|
// |        |                |           |               |               |                   |               |           |                 |
// |        | t1 -> [e0, e4] |  e0: t1   |     e0: t1    |    e0: t1     |  [t1, t6] -> e0   |    e0: t1     |  e0: t1   | [e0, e4] ->  t1 |
// |        | t2 -> [e2, e5] |  e1:      |     e1:       |    e0:        |  [t3, t7] -> e1   |    e1:        |  e1:      | [e2, e5] ->  t2 |
// |        |                |  e2: t2   |     e0:       |    e0: t6     |                   |    e0:        |  e2: t2   |                 |
// |   d0   |                |  e3:      |     e1: t3    |    e0:        |                   |    e1: t3     |  e3:      |                 |
// |        |                |  e4: t1   |     e0: t6    |    e1:        |                   |    e0: t6     |  e4: t1   |                 |
// |        |                |  e5: t2   |     e1:       |    e1: t3     |                   |    e1:        |  e5: t2   |                 |
// |        |                |  e6:      |     e0:       |    e1:        |                   |    e0:        |  e6:      |                 |
// |        |                |  e7:      |     e1: t7    |    e1: t7     |                   |    e1: t7     |  e7:      |                 |
// |________|________________|___________|_______________|_______________|___________________|_______________|___________|_________________|
// |        |                |           |               |               |                   |               |           |                 |
// |        | t3 -> [e1, e6] |  e0:      |     e2: t2    |    e2: t2     |  [t2, t8] -> e2   |    e2: t2     |  e0:      | [e1, e6] ->  t3 |
// |        | t4 -> [e3, e4] |  e1: t3   |     e3:       |    e2:        |  [t4, t5] -> e3   |    e3:        |  e1: t3   | [e3, e4] ->  t4 |
// |        |                |  e2:      |     e2:       |    e2:        |                   |    e2:        |  e2:      |                 |
// |   d1   |                |  e3: t4   |     e3: t4    |    e2: t8     |                   |    e3: t4     |  e3: t4   |                 |
// |        |                |  e4: t4   |     e2:       |    e3:        |                   |    e2:        |  e4: t4   |                 |
// |        |                |  e5:      |     e3: t5    |    e3: t4     |                   |    e3: t5     |  e5:      |                 |
// |        |                |  e6: t3   |     e2: t8    |    e3: t5     |                   |    e2: t8     |  e6: t3   |                 |
// |        |                |  e7:      |     e3:       |    e3:        |                   |    e3:        |  e7:      |                 |
// |________|________________|___________|_______________|_______________|___________________|_______________|___________|_________________|
// |        |                |           |               |               |                   |               |           |                 |
// |        | t5 -> [e3, e7] |  e0: t6   |     e4: t1    |    e4: t1     |  [t1,t4,t6] -> e4 |    e4: t1     |  e0: t6   | [e3, e7] ->  t5 |
// |        | t6 -> [e0, e4] |  e1:      |     e5: t2    |    e4: t4     |  [t2, t8]   -> e5 |    e5: t2     |  e1:      | [e0, e4] ->  t6 |
// |        |                |  e2:      |     e4: t4    |    e4: t6     |                   |    e4: t4     |  e2:      |                 |
// |   d2   |                |  e3: t5   |     e5:       |    e4:        |                   |    e5:        |  e3: t5   |                 |
// |        |                |  e4: t6   |     e4: t6    |    e5: t2     |                   |    e4: t6     |  e4: t6   |                 |
// |        |                |  e5:      |     e5:       |    e5:        |                   |    e5:        |  e5:      |                 |
// |        |                |  e6:      |     e4:       |    e5:        |                   |    e4:        |  e6:      |                 |
// |        |                |  e7: t5   |     e5: t8    |    e5: t8     |                   |    e5: t8     |  e7: t5   |                 |
// |________|________________|___________|_______________|_______________|___________________|_______________|___________|_________________|
// |        |                |           |               |               |                   |               |           |                 |
// |        | t7 -> [e1, e7] |  e0:      |     e6:       |    e6:        |  [t3]     -> e6   |    e6:        |  e0:      | [e1, e7] ->  t7 |
// |        | t8 -> [e2, e5] |  e1: t7   |     e7:       |    e6: t3     |  [t5, t7] -> e7   |    e7:        |  e1: t7   | [e2, e5] ->  t8 |
// |        |                |  e2: t8   |     e6: t3    |    e6:        |                   |    e6: t3     |  e2: t8   |                 |
// |   d3   |                |  e3:      |     e7:       |    e6:        |                   |    e7:        |  e3:      |                 |
// |        |                |  e4:      |     e6:       |    e7:        |                   |    e6:        |  e4:      |                 |
// |        |                |  e5: t8   |     e7: t5    |    e7:        |                   |    e7: t5     |  e5: t8   |                 |
// |        |                |  e6:      |     e6:       |    e7: t5     |                   |    e6:        |  e6:      |                 |
// |        |                |  e7: t7   |     e7: t7    |    e7: t7     |                   |    e7: t7     |  e7: t7   |                 |
// |________|________________|___________|_______________|_______________|___________________|_______________|___________|_________________|
// clang-format on

AlltoAllTokenDispatcher::AlltoAllTokenDispatcher(int64_t n_experts,
                                                 int64_t n_local_experts,
                                                 ProcessGroup* ep_pg)
    : n_experts_(n_experts), n_local_experts_(n_local_experts), ep_pg_(ep_pg) {
  CHECK_EQ(n_experts % n_local_experts, 0)
      << "n_experts should be divisible by n_local_experts";
  ep_size_ = ep_pg_->world_size();
  ep_rank_ = ep_pg_->rank();

  // [n_experts]: [0, 1, 2, 3, 4, 5, 6, 7, ...]
  auto chunk_idxs = torch::arange(n_experts);
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
      to_vector(chunk_idxs.reshape({-1, n_local_experts}).t().ravel());

  // [n_experts] => [ep_size * n_local_experts]
  // => [n_local_experts, ep_size]
  // => [ep_size, n_local_experts]
  // => [ep_size*n_local_experts]
  // for example: n_experts = 8, n_local_experts = 2, ep_size = 4
  // [0, 1, 2, 3, 4, 5, 6, 7]
  // => [[0, 1, 2, 3], [4, 5, 6, 7]]
  // => [[0, 4], [1, 5], [2, 6], [3, 7]]
  // => [0, 4, 1, 5, 2, 6, 3, 7]
  this->restore_output_by_local_experts_ =
      to_vector(chunk_idxs.reshape({n_local_experts, -1}).t().ravel());
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
      to_vector(local_tokens_per_expert.reshape({ep_size_, -1}).sum(/*dim=*/1));

  // gather the global distribution of tokens accross ranks
  // [n_experts] => [ep_size, n_experts]
  auto tokens_per_expert = allgather(local_tokens_per_expert, ep_pg_);

  // slice tokens for local experts
  // [ep_size, n_local_experts]
  auto tokens_per_local_expert =
      tokens_per_expert.slice(/*dim=*/1,
                              /*start=*/n_local_experts_ * ep_rank_,
                              /*end=*/n_local_experts_ * (ep_rank_ + 1));
  // number of tokens from each rank
  // [ep_size, n_local_experts] => [ep_size]
  this->output_splits_ = to_vector(tokens_per_local_expert.sum(/*dim=*/1));

  // [ep_size*n_local_experts]
  this->tokens_per_local_expert_ = to_vector(tokens_per_local_expert.ravel());
  // [n_local_experts*ep_size]
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

  // alltoall communication to gather tokens for local experts from all ranks
  // sorted by (ep_size, n_local_experts)
  auto permuted_tokens = alltoall(
      local_permuted_tokens, this->input_splits_, this->output_splits_, ep_pg_);

  if (n_local_experts_ > 1) {
    // sort tokens by (n_local_experts, ep_size)
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
  // permuted_tokens sorted by (n_local_experts, ep_size)
  if (n_local_experts_ > 1) {
    // sort tokens by (ep_size, n_local_experts)
    permuted_tokens = sort_by_idxs(permuted_tokens,
                                   this->restore_tokens_per_local_expert_,
                                   this->restore_output_by_local_experts_);
  }
  // alltoall communication to gather local tokens back from all ranks
  auto local_permuted_tokens =
      alltoall(permuted_tokens,
               /*input_split_sizes=*/this->output_splits_,
               /*output_split_sizes=*/this->input_splits_,
               ep_pg_);

  // apply weights for each expert
  // [n_permuted_tokens, dim] * [n_permuted_tokens]
  local_permuted_tokens *= this->permuted_probs_.unsqueeze(/*dim=*/-1);

  return unpermute(
      local_permuted_tokens, this->sorted_indices_, this->restore_shape_);
}

}  // namespace llm