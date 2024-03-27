#include "sampler.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include "sampling/parameters.h"
namespace llm {

Sampler::Sampler(const SamplingParameters& params) {
  sample_funcs_.reserve(params.do_sample.size());
  for (bool sample : params.do_sample) {
    // choose right sample function for each sequence
    auto func = sample ? random_sample : greedy_sample;
    sample_funcs_.emplace_back(func);
  }
}

torch::Tensor Sampler::sample(const torch::Tensor& probs) const {
  const auto num_seqs = probs.size(/*dim=*/0);
  auto selected = torch::empty(
      {num_seqs, 1}, torch::dtype(torch::kInt64).device(probs.device()));
  // sample logits for each sequence
  for (int64_t i = 0; i < num_seqs; ++i) {
    // Sample from the adjusted distribution
    auto sample = sample_funcs_[i](probs[i]);
    selected.index_put_({i}, sample);
  }
  return selected;
}

SampleOutput Sampler::forward(const torch::Tensor& logits) const {
  const auto num_seqs = logits.size(0);
  CHECK_EQ(num_seqs, static_cast<int64_t>(sample_funcs_.size()));

  // use float32 for probabilities and log probabilities
  const auto probs =
      torch::softmax(logits, /*dim=*/-1, /*dtype=*/torch::kFloat32);
  const auto logprobs =
      torch::log_softmax(logits, /*dim=*/-1, /*dtype=*/torch::kFloat32);

  SampleOutput output;
  output.next_tokens = sample(probs);
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
