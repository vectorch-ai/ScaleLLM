#include "sampler.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include "sampling/parameters.h"
namespace llm {

Sampler::Sampler(const torch::Tensor& do_sample) {
  CHECK(do_sample.defined());
  do_sample_ = do_sample;
  all_random_sample_ = do_sample.all().item<bool>();
  all_greedy_sample_ = !do_sample.any().item<bool>();
}

SampleOutput Sampler::forward(const torch::Tensor& logits) const {
  // same batch size
  CHECK_EQ(logits.size(0), do_sample_.size(0));

  // use float32 for probabilities and log probabilities
  const auto probs =
      torch::softmax(logits, /*dim=*/-1, /*dtype=*/torch::kFloat32);
  const auto logprobs =
      torch::log_softmax(logits, /*dim=*/-1, /*dtype=*/torch::kFloat32);

  SampleOutput output;
  output.probs = probs;
  output.logprobs = logprobs;

  if (all_random_sample_) {
    output.next_tokens = random_sample(probs);
  } else if (all_greedy_sample_) {
    output.next_tokens = greedy_sample(probs);
  } else {
    // mixed sample, sample both then choose based on do_sample_
    auto random = random_sample(probs);
    auto greedy = greedy_sample(probs);
    output.next_tokens = torch::where(do_sample_, random, greedy);
  }

  return output;
}

torch::Tensor Sampler::greedy_sample(const torch::Tensor& probs) {
  return probs.argmax(/*dim=*/-1);
}

torch::Tensor Sampler::random_sample(const torch::Tensor& probs) {
  // return probs.multinomial(/*num_samples=*/1, /*replacement=*/false);
  // Avoid the expensive GPU<->CPU sync done by torch::multinomial
  auto q = torch::empty_like(probs).exponential_(/*lambd=*/1);
  return probs.div(q).argmax(/*dim=*/-1);
}

}  // namespace llm
