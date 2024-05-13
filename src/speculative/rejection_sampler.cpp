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

RejectionSampler::RejectionSampler(const torch::Tensor& do_sample) {
  // [batch_size, 1]
  do_sample_ = do_sample.unsqueeze_(/*dim=*/-1);
  all_random_sample_ = do_sample.all().item<bool>();
  all_greedy_sample_ = !do_sample.any().item<bool>();
}

// draft_token_ids: [batch_size, n_speculative_tokens]
// draft_probs: [batch_size, n_speculative_tokens, vocab_size]
// target_probs: [batch_size, n_speculative_tokens, vocab_size]
// bonus_token_ids: [batch_size, 1]
// returns accepted tokens. [batch_size, n_speculative_tokens + 1]
torch::Tensor RejectionSampler::forward(const torch::Tensor& draft_token_ids,
                                        const torch::Tensor& draft_probs,
                                        const torch::Tensor& target_probs,
                                        const torch::Tensor& bonus_token_ids,
                                        bool mask_out_rejected_tokens) const {
  CHECK_EQ(draft_token_ids.size(0), do_sample_.size(0))
      << "batch size mismatch";
  DCHECK_EQ(draft_token_ids.size(1), draft_probs.size(1));
  DCHECK_EQ(draft_probs.sizes(), target_probs.sizes());

  if (all_greedy_sample_) {
    return greedy_sample(draft_token_ids,
                         target_probs,
                         bonus_token_ids,
                         mask_out_rejected_tokens);
  }

  auto uniform_rand =
      torch::rand(draft_token_ids.sizes(), draft_probs.options());
  if (all_random_sample_) {
    return random_sample(draft_token_ids,
                         draft_probs,
                         target_probs,
                         uniform_rand,
                         bonus_token_ids,
                         mask_out_rejected_tokens);
  }

  // mixed sample, sample both then choose based on do_sample_
  auto random = random_sample(draft_token_ids,
                              draft_probs,
                              target_probs,
                              uniform_rand,
                              bonus_token_ids,
                              mask_out_rejected_tokens);
  auto greedy = greedy_sample(
      draft_token_ids, target_probs, bonus_token_ids, mask_out_rejected_tokens);
  return torch::where(do_sample_, random, greedy);
}

// build mask from accepted matrix
// for example: [[1, 1, 0, 1],   ->   [[1, 1, 1, 0, 0],
//               [1, 0, 0, 0]]         [1, 1, 0, 0, 0]]
torch::Tensor RejectionSampler::build_accepted_mask(
    const torch::Tensor& accepted) {
  // build the mask for the first rejected token
  const auto batch_size = accepted.size(0);
  const auto n_tokens = accepted.size(1);

  // use LongTensor since argmax does not support bool
  auto accepted_int64 = accepted.to(torch::kInt64);
  auto bonus_mask = torch::zeros({batch_size, 1}, accepted_int64.options());
  auto combined_mask = torch::cat({accepted_int64, bonus_mask}, /*dim=*/-1);
  // [batch_size, 1]
  auto first_rejected_mask =
      (1 - combined_mask).argmax(/*dim=*/1, /*keepdim=*/true);

  // [1, n_speculative_tokens + 1]
  auto indices =
      torch::arange(n_tokens + 1, accepted.device()).unsqueeze(/*dim=*/0);
  // [batch_size, n_speculative_tokens + 1]
  auto accepted_mask = indices <= first_rejected_mask;
  return accepted_mask;
}

torch::Tensor RejectionSampler::random_sample(
    const torch::Tensor& draft_token_ids,
    const torch::Tensor& draft_probs,
    const torch::Tensor& target_probs,
    const torch::Tensor& uniform_rand,
    const torch::Tensor& bonus_token_ids,
    bool mask_out_rejected_tokens) {
  auto selected_draft_probs =
      index_select_2d(draft_probs, /*dim=*/-1, draft_token_ids);
  auto selected_target_probs =
      index_select_2d(target_probs, /*dim=*/-1, draft_token_ids);

  // std::min(probs, 1.0) element-wise
  auto acceptance_probs = (selected_target_probs / selected_draft_probs);
  auto accepted = (uniform_rand < acceptance_probs);

  // construct recovered probs
  const auto epsilon = std::numeric_limits<float>::epsilon();
  auto recovered_probs = (target_probs - draft_probs);
  // a small value to avoid division by zero
  recovered_probs.clamp_min_(/*min=*/epsilon);

  auto recovered_probs_sum = recovered_probs.sum(-1, /*keepdim=*/true);
  recovered_probs.div_(recovered_probs_sum);

  // resample on the recovered probs
  torch::Tensor recovered_token_ids = Sampler::random_sample(recovered_probs);

  auto combined = torch::where(accepted, draft_token_ids, recovered_token_ids);
  // [batch_size, n_speculative_tokens + 1]
  auto accepted_token_ids = torch::cat({combined, bonus_token_ids}, /*dim=*/-1);

  if (mask_out_rejected_tokens) {
    // build the mask for the first rejected token
    auto accepted_mask = build_accepted_mask(accepted);
    // mask out the rejected tokens with -1
    accepted_token_ids = torch::where(accepted_mask,
                                      accepted_token_ids,
                                      -torch::ones_like(accepted_token_ids));
  }

  return accepted_token_ids;
}

torch::Tensor RejectionSampler::greedy_sample(
    const torch::Tensor& draft_token_ids,
    const torch::Tensor& target_probs,
    const torch::Tensor& bonus_token_ids,
    bool mask_out_rejected_tokens) {
  auto target_token_ids = Sampler::greedy_sample(target_probs);

  // mask out the rejected tokens with -1
  // [batch_size, n_speculative_tokens + 1]
  auto accepted_token_ids =
      torch::cat({target_token_ids, bonus_token_ids}, /*dim=*/-1);

  if (mask_out_rejected_tokens) {
    // [batch_size, n_speculative_tokens + 1]
    auto accepted = (target_token_ids == draft_token_ids);
    auto accepted_mask = build_accepted_mask(accepted);
    // mask out the rejected tokens with -1
    accepted_token_ids = torch::where(accepted_mask,
                                      accepted_token_ids,
                                      -torch::ones_like(accepted_token_ids));
  }
  return accepted_token_ids;
}

}  // namespace llm
