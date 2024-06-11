#include "rejection_sampler.h"

#include <ATen/ops/stack.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include "sampling/parameters.h"
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
SampleOutput RejectionSampler::forward(const torch::Tensor& draft_token_ids,
                                       const torch::Tensor& draft_probs,
                                       const torch::Tensor& target_logits,
                                       const torch::Tensor& bonus_token_ids) const {
  CHECK_EQ(draft_token_ids.size(0), do_sample_.size(0))
      << "batch size mismatch";
  DCHECK_EQ(draft_token_ids.size(1), draft_probs.size(1));
  // DCHECK_EQ(draft_probs.sizes(), target_probs.sizes());

  auto target_probs =
      torch::softmax(target_logits, /*dim=*/-1, /*dtype=*/torch::kFloat32);
  // filter out probs for bonus tokens
  target_probs = target_probs.slice(
      /*dim=*/1, /*start=*/0, /*end=*/target_probs.size(1) - 1);

  torch::Tensor target_token_ids;
  if (all_greedy_sample_) {
    target_token_ids = greedy_sample(target_probs);
  } else if (all_random_sample_) {
    const auto uniform_rand =
        torch::rand(draft_token_ids.sizes(), draft_probs.options());
    target_token_ids =
        random_sample(draft_token_ids, draft_probs, target_probs, uniform_rand);
  } else {
    const auto uniform_rand =
        torch::rand(draft_token_ids.sizes(), draft_probs.options());
    // mixed sample, sample both then choose based on do_sample_
    auto random =
        random_sample(draft_token_ids, draft_probs, target_probs, uniform_rand);
    auto greedy = greedy_sample(target_probs);
    target_token_ids = torch::where(do_sample_, random, greedy);
  }

  SampleOutput output;

  // [batch_size, n_speculative_tokens + 1]
  const auto accepted_token_ids =
      torch::cat({target_token_ids, bonus_token_ids}, /*dim=*/-1);
  output.next_tokens = accepted_token_ids;
  bool logprobs_ = true;
  int64_t top_logprobs_ = 2;
  if (logprobs_) {
    // log_softmax is equivalent to log(softmax) but more numerically stable
    auto target_logprobs = torch::log_softmax(
        target_logits, /*dim=*/-1, /*dtype=*/torch::kFloat32);
    
    // select the logprobs for each sequence
    output.logprobs = target_logprobs.gather(/*dim=*/-1, accepted_token_ids);

    if (top_logprobs_ > 0) {
      auto [values, indices] = target_logprobs.topk(top_logprobs_, /*dim=*/-1);
      output.top_logprobs = values;
      output.top_tokens = indices;
    }
  }
  return output;
}

torch::Tensor RejectionSampler::random_sample(
    const torch::Tensor& draft_token_ids,
    const torch::Tensor& draft_probs,
    const torch::Tensor& target_probs,
    const torch::Tensor& uniform_rand) {
  auto selected_draft_probs =
      index_select_2d(draft_probs, /*dim=*/-1, draft_token_ids);
  auto selected_target_probs =
      index_select_2d(target_probs, /*dim=*/-1, draft_token_ids);

  // std::min(probs, 1.0) element-wise
  auto acceptance_probs = (selected_target_probs / selected_draft_probs);
  auto accepted = (uniform_rand < acceptance_probs);

  // construct recovered probs
  auto recovered_probs = (target_probs - draft_probs).clamp_min_(0);
  // a small value to avoid division by zero
  const auto epsilon = 1e-6f;
  auto sum = recovered_probs.sum(-1, /*keepdim=*/true).clamp_min_(epsilon);
  recovered_probs.div_(sum);

  // resample on the recovered probs
  torch::Tensor recovered_token_ids = Sampler::random_sample(recovered_probs);
  return torch::where(accepted, draft_token_ids, recovered_token_ids);
}

torch::Tensor RejectionSampler::greedy_sample(
    const torch::Tensor& target_probs) {
  return Sampler::greedy_sample(target_probs);
}

}  // namespace llm
