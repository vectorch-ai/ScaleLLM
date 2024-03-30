#include "rejection_sampler.h"

#include <ATen/ops/stack.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include "sampling/sampler.h"

namespace llm {

namespace {
// index_select that supports multiple dimensions index
torch::Tensor index_select_2d(const torch::Tensor& input,
                              int64_t dim,
                              const torch::Tensor& index) {
  return input.gather(dim, index.unsqueeze(dim)).squeeze(dim);
}

}  // namespace

RejectionSampler::RejectionSampler(const std::vector<bool>& do_sample,
                                   const torch::TensorOptions& options) {
  std::vector<int32_t> do_sample_int;
  do_sample_int.reserve(do_sample.size());
  for (bool sample : do_sample) {
    if (sample) {
      all_greedy_sample_ = false;
      do_sample_int.push_back(1);
    } else {
      all_random_sample_ = false;
      do_sample_int.push_back(0);
    }
  }
  // [batch_size, 1]
  do_sample_ = torch::tensor(do_sample_int, options.dtype(torch::kBool))
                   .unsqueeze_(/*dim=*/-1);
}

// draft_token_ids: [batch_size, n_speculative_tokens]
// draft_probs: [batch_size, n_speculative_tokens, vocab_size]
// target_probs: [batch_size, n_speculative_tokens, vocab_size]
// bonus_token_ids: [batch_size, 1]
// returns accepted tokens. [batch_size, n_speculative_tokens + 1]
torch::Tensor RejectionSampler::forward(
    const torch::Tensor& draft_token_ids,
    const torch::Tensor& draft_probs,
    const torch::Tensor& target_probs,
    const torch::Tensor& bonus_token_ids) const {
  DCHECK(draft_token_ids.size(1) == draft_probs.size(1));
  DCHECK(draft_probs.sizes() == target_probs.sizes());

  // [batch_size, n_speculative_tokens]
  torch::Tensor accepted_token_ids;
  if (all_random_sample_) {
    accepted_token_ids =
        batch_random_sample(draft_token_ids, draft_probs, target_probs);
  } else if (all_greedy_sample_) {
    accepted_token_ids =
        batch_greedy_sample(draft_token_ids, draft_probs, target_probs);
  } else {
    // mixed sample, sample both then choose based on do_sample_
    auto random =
        batch_random_sample(draft_token_ids, draft_probs, target_probs);
    auto greedy =
        batch_greedy_sample(draft_token_ids, draft_probs, target_probs);
    accepted_token_ids = torch::where(do_sample_, random, greedy);
  }

  // [batch_size, n_speculative_tokens + 1]
  return torch::cat({accepted_token_ids, bonus_token_ids}, /*dim=*/-1);
}

torch::Tensor RejectionSampler::batch_random_sample(
    const torch::Tensor& draft_token_ids,
    const torch::Tensor& draft_probs,
    const torch::Tensor& target_probs) {
  const auto batch_size = draft_token_ids.size(0);
  const auto n_tokens = draft_token_ids.size(1);
  auto selected_draft_probs =
      index_select_2d(draft_probs, /*dim=*/-1, draft_token_ids);
  auto selected_target_probs =
      index_select_2d(target_probs, /*dim=*/-1, draft_token_ids);

  auto uniform_rand =
      torch::rand({batch_size, n_tokens}, draft_probs.options());
  auto acceptance_probs =
      (selected_target_probs / selected_draft_probs).clamp_max_(1.0f);
  auto accepted = uniform_rand < acceptance_probs;

  // construct recovered probs
  const auto eps = 1e-6f;
  auto recovered_probs = target_probs - draft_probs;
  auto recovered_probs_sum =
      recovered_probs.sum(-1, /*keepdim=*/true).clamp_min_(eps);
  recovered_probs.div_(recovered_probs_sum);
  // resample on the recovered probs
  auto recovered_token_ids = Sampler::random_sample(recovered_probs);
  return torch::where(accepted, draft_token_ids, recovered_token_ids);
}

torch::Tensor RejectionSampler::batch_greedy_sample(
    const torch::Tensor& /*draft_token_ids*/,
    const torch::Tensor& /*draft_probs*/,
    const torch::Tensor& target_probs) {
  return Sampler::greedy_sample(target_probs);
}

}  // namespace llm
