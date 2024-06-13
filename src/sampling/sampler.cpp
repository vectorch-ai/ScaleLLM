#include "sampler.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include "sampling/parameters.h"
namespace llm {

Sampler::Sampler(const torch::Tensor& do_sample,
                 bool logprobs,
                 int64_t max_top_logprobs)
    : logprobs_(logprobs), max_top_logprobs_(max_top_logprobs) {
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

  SampleOutput output;
  output.probs = probs;

  torch::Tensor samples;
  if (all_random_sample_) {
    samples = random_sample(probs);
  } else if (all_greedy_sample_) {
    samples = greedy_sample(probs);
  } else {
    // mixed sample, sample both then choose based on do_sample_
    auto random = random_sample(probs);
    auto greedy = greedy_sample(probs);
    samples = torch::where(do_sample_, random, greedy);
  }
  output.next_tokens = samples;

  if (logprobs_) {
    // log_softmax is equivalent to log(softmax) but more numerically stable
    const auto logprobs =
        torch::log_softmax(logits, /*dim=*/-1, /*dtype=*/torch::kFloat32);
    // select the logprobs for each sequence
    auto selected_logprobs = logprobs.gather(/*dim=*/-1, samples.view({-1, 1}));
    output.logprobs = selected_logprobs.view({-1});

    if (max_top_logprobs_ > 0) {
      auto [values, indices] = logprobs.topk(max_top_logprobs_, /*dim=*/-1);
      output.top_logprobs = values;
      output.top_tokens = indices;
    }
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
