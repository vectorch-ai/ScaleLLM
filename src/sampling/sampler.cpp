#include "sampler.h"

#include <glog/logging.h>
#include <torch/torch.h>
namespace llm {
namespace {
torch::Tensor greedy_sample(const torch::Tensor& probs) {
  return probs.argmax(/*dim=*/-1);
}

torch::Tensor multinomial_sample(const torch::Tensor& probs) {
  return probs.multinomial(/*num_samples=*/1, /*replacement=*/false);
}

}  // namespace

Sampler::Sampler(const SamplingParameters& params,
                 torch::ScalarType dtype,
                 const torch::Device& device) {
  // initialize top_p if any of the values are not 1.0
  if (std::any_of(params.top_p.begin(), params.top_p.end(), [](float t) {
        return t != 1.0;
      })) {
    top_p_ = torch::tensor(params.top_p, torch::dtype(dtype).device(device))
                 .unsqueeze(1);
  }

  // initialize top_k if any of the values are not 0
  if (std::any_of(params.top_k.begin(), params.top_k.end(), [](int64_t t) {
        return t != 0;
      })) {
    top_k_ =
        torch::tensor(params.top_k, torch::dtype(torch::kLong).device(device))
            .unsqueeze(1);
    // Replace 0 with max_value to disable top_k
    const auto max_value = std::numeric_limits<int64_t>::max();
    top_k_ = torch::where(top_k_ == 0, torch::tensor(max_value), top_k_);
  }

  sample_funcs_.reserve(params.do_sample.size());
  for (bool sample : params.do_sample) {
    // choose right sample function for each sequence
    auto func = sample ? multinomial_sample : greedy_sample;
    sample_funcs_.emplace_back(func);
  }
}

torch::Tensor Sampler::sample(const torch::Tensor& probs) const {
  const auto num_seqs = probs.size(0);
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

torch::Tensor Sampler::forward(const torch::Tensor& logits) const {
  const auto num_seqs = logits.size(0);
  CHECK_EQ(num_seqs, static_cast<int64_t>(sample_funcs_.size()));

  const auto probs = torch::softmax(logits, /*dim=*/-1);
  if (!top_p_.defined() && !top_k_.defined()) {
    // No top_p or top_k, just sample from the distribution
    return sample(probs);
  }

  auto [probs_sort, probs_idx] = probs.sort(/*dim=*/-1, /*descending=*/true);
  // ####################  apply top p   ####################
  if (top_p_.defined()) {
    // Calculate the cumulative sum of sorted probabilities
    const auto probs_sum = torch::cumsum(probs_sort, /*dim=*/-1);
    // Create a mask where (cumulative sum - current value) > p
    const auto mask = (probs_sum - probs_sort) > top_p_;
    // Set values where mask is true to 0.0
    probs_sort.masked_fill_(mask, 0.0);
  }

  // ####################  apply top k   ####################
  if (top_k_.defined()) {
    const auto vocab_size = logits.size(-1);
    auto top_k_mask = torch::arange(vocab_size, probs_sort.device())
                          .expand(probs_sort.sizes());
    top_k_mask = top_k_mask >= top_k_;
    // mask fill the values that are not in the top k
    probs_sort.masked_fill_(top_k_mask, 0.0);
  }

  // Adjust the probability of the selected tokens
  probs_sort.div_(probs_sort.sum(-1, /*keepdim=*/true));

  // Sample from the adjusted distribution
  const auto selected = sample(probs_sort);
  // Get the original indices of the sampled values
  return torch::gather(probs_idx, /*dim=*/-1, selected);
}

}  // namespace llm
