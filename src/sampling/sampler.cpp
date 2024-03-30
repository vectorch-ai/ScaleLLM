#include "sampler.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include "sampling/parameters.h"
namespace llm {

Sampler::Sampler(const std::vector<bool>& do_sample) : do_sample_(do_sample) {
  for (bool sample : do_sample_) {
    if (sample) {
      all_greedy_sample_ = false;
    } else {
      all_random_sample_ = false;
    }
  }
}

torch::Tensor Sampler::sample(const torch::Tensor& probs) const {
  const auto num_seqs = probs.size(/*dim=*/0);
  auto selected = torch::empty(
      {num_seqs, 1}, torch::dtype(torch::kInt64).device(probs.device()));
  // sample sequence one by one
  for (int64_t i = 0; i < num_seqs; ++i) {
    if (do_sample_[i]) {
      selected[i] = random_sample(probs[i]);
    } else {
      selected[i] = greedy_sample(probs[i]);
    }
  }
  return selected;
}

SampleOutput Sampler::forward(const torch::Tensor& logits) const {
  const auto num_seqs = logits.size(0);
  CHECK_EQ(num_seqs, static_cast<int64_t>(do_sample_.size()));

  // use float32 for probabilities and log probabilities
  const auto probs =
      torch::softmax(logits, /*dim=*/-1, /*dtype=*/torch::kFloat32);
  const auto logprobs =
      torch::log_softmax(logits, /*dim=*/-1, /*dtype=*/torch::kFloat32);

  SampleOutput output;
  if (all_greedy_sample_) {
    output.next_tokens = greedy_sample(probs);
  } else if (all_random_sample_) {
    output.next_tokens = random_sample(probs);
  } else {
    output.next_tokens = sample(probs);
  }
  // TODO: add logprobs to output
  return output;
}

torch::Tensor Sampler::greedy_sample(const torch::Tensor& probs) {
  return probs.argmax(/*dim=*/-1);
}

torch::Tensor Sampler::random_sample(const torch::Tensor& probs) {
  // return probs.multinomial(/*num_samples=*/1, /*replacement=*/false);
  // Avoid the expensive GPU<->CPU sync done by torch::multinomial
  auto q = torch::empty_like(probs).exponential_(/*lambd=*/1);
  return probs.div_(q).argmax(/*dim=*/-1);
}

}  // namespace llm
