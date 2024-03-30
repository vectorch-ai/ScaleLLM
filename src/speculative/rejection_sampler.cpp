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

RejectionSampler::RejectionSampler(const std::vector<bool>& do_sample)
    : do_sample_(do_sample) {
  for (bool sample : do_sample_) {
    if (sample) {
      all_greedy_sample_ = false;
    } else {
      all_random_sample_ = false;
    }
  }
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

  torch::Tensor accepted_token_ids;
  if (all_random_sample_) {
    accepted_token_ids =
        batch_random_sample(draft_token_ids, draft_probs, target_probs);
  } else if (all_greedy_sample_) {
    accepted_token_ids =
        batch_greedy_sample(draft_token_ids, draft_probs, target_probs);
  } else {
    // mixed sample, sample one by one
    accepted_token_ids =
        rejection_sample(draft_token_ids, draft_probs, target_probs);
  }
  return torch::cat({accepted_token_ids, bonus_token_ids}, /*dim=*/-1);
}

// rejection sample sequence one by one
torch::Tensor RejectionSampler::rejection_sample(
    const torch::Tensor& draft_token_ids,
    const torch::Tensor& draft_probs,
    const torch::Tensor& target_probs) const {
  const auto seq_len = draft_token_ids.size(0);

  auto selected_draft_probs =
      draft_probs.index_select(/*dim=*/0, /*index=*/draft_token_ids);
  auto selected_target_probs =
      target_probs.index_select(/*dim=*/0, /*index=*/draft_token_ids);

  // construct recovered probs
  const auto eps = 1e-6f;
  // add a small epsilon to avoid division by zero
  auto recovered_probs = (target_probs - draft_probs).clamp_min_(eps);
  recovered_probs.div_(recovered_probs.sum(-1, /*keepdim=*/true));

  auto output = torch::empty_like(draft_token_ids);
  for (int64_t i = 0; i < seq_len; ++i) {
    bool sample = do_sample_[i];
    if (sample) {
      auto uniform_rand = torch::rand({1}, draft_probs.options());
      auto acceptance_probs =
          selected_target_probs[i] / selected_draft_probs[i];
      // clamp the acceptance_probs
      acceptance_probs.clamp_max_(1.0);
      auto accepted = (uniform_rand < acceptance_probs);
      if (accepted.item<bool>()) {
        output[i] = draft_token_ids[i];
      } else {
        output[i] = Sampler::random_sample(recovered_probs[i]);
      }
    } else {
      output[i] = Sampler::greedy_sample(target_probs[i]);
    }
  }
  return output;
}

// draft_token_ids: [batch_size, n_speculative_tokens]
// draft_probs: [batch_size, n_speculative_tokens, vocab_size]
// target_probs: [batch_size, n_speculative_tokens, vocab_size]
// bonus_token_ids: [batch_size, 1]

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
  auto acceptance_probs = selected_target_probs / selected_draft_probs;
  // clamp the acceptance_probs
  acceptance_probs.clamp_max_(1.0f);
  auto accepted = uniform_rand < acceptance_probs;

  // construct recovered probs
  const auto eps = 1e-6f;
  auto recovered_probs = (target_probs - draft_probs).clamp_min_(eps);
  recovered_probs.div_(recovered_probs.sum(-1, /*keepdim=*/true));
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
