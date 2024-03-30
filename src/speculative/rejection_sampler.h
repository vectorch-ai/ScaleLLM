#pragma once
#include <c10/core/TensorOptions.h>
#include <torch/torch.h>
#include <torch/types.h>

#include <functional>
#include <optional>

#include "sampling/parameters.h"
namespace llm {

class RejectionSampler final {
 public:
  RejectionSampler(const std::vector<bool>& do_sample);

  // operator() allows us to use the module as a function.
  template <typename... Args>
  auto operator()(Args&&... args) const {
    return this->forward(::std::forward<Args>(args)...);
  }

  // Sample tokens ids using rejection sampling.
  // draft_token_ids: [batch_size, n_speculative_tokens]
  // draft_probs: [batch_size, n_speculative_tokens, vocab_size]
  // target_probs: [batch_size, n_speculative_tokens, vocab_size]
  // bonus_token_ids: [batch_size, 1]
  // returns accepted tokens. [batch_size, n_speculative_tokens + 1]
  // N.B. caller needs to compare accepted_token_ids with draft_token_ids to
  // filter out rejected tokens (after the first mismatched token).
  torch::Tensor forward(const torch::Tensor& draft_token_ids,
                        const torch::Tensor& draft_probs,
                        const torch::Tensor& target_probs,
                        const torch::Tensor& bonus_token_ids) const;

 private:
  // rejection sample one sequence by one sequence
  torch::Tensor rejection_sample(const torch::Tensor& draft_token_ids,
                                 const torch::Tensor& draft_probs,
                                 const torch::Tensor& target_probs) const;

  // batch random rejection sample
  static torch::Tensor batch_random_sample(const torch::Tensor& draft_token_ids,
                                           const torch::Tensor& draft_probs,
                                           const torch::Tensor& target_probs);

  // batch greedy rejection sample
  static torch::Tensor batch_greedy_sample(const torch::Tensor& draft_token_ids,
                                           const torch::Tensor& draft_probs,
                                           const torch::Tensor& target_probs);

  std::vector<bool> do_sample_;
  bool all_random_sample_ = true;
  bool all_greedy_sample_ = true;
};

}  // namespace llm
