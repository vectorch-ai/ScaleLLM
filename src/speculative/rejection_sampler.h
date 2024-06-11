#pragma once
#include <torch/torch.h>
#include <torch/types.h>

#include "sampling/sampler.h"
namespace llm {

class RejectionSampler final {
 public:
  RejectionSampler(const torch::Tensor& do_sample);

  // operator() allows us to use the module as a function.
  template <typename... Args>
  auto operator()(Args&&... args) const {
    return this->forward(::std::forward<Args>(args)...);
  }

  // Sample tokens ids using rejection sampling.
  // draft_token_ids: [batch_size, n_speculative_tokens]
  // draft_probs: [batch_size, n_speculative_tokens, vocab_size]
  // target_logits: [batch_size, n_speculative_tokens + 1, vocab_size]
  // bonus_token_ids: [batch_size, 1]
  SampleOutput forward(const torch::Tensor& draft_token_ids,
                       const torch::Tensor& draft_probs,
                       const torch::Tensor& target_logits,
                       const torch::Tensor& bonus_token_ids) const;

  static torch::Tensor random_sample(const torch::Tensor& draft_token_ids,
                                     const torch::Tensor& draft_probs,
                                     const torch::Tensor& target_probs,
                                     const torch::Tensor& uniform_rand);

  static torch::Tensor greedy_sample(const torch::Tensor& target_probs);

 private:
  // [batch_size]
  torch::Tensor do_sample_;
  bool all_random_sample_ = true;
  bool all_greedy_sample_ = true;
};

}  // namespace llm
