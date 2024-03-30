#pragma once
#include <c10/core/TensorOptions.h>
#include <torch/torch.h>
#include <torch/types.h>

#include <functional>

#include "sampling/parameters.h"
namespace llm {

class Sampler final {
 public:
  Sampler(const std::vector<bool>& do_sample);

  // operator() allows us to use the module as a function.
  template <typename... Args>
  auto operator()(Args&&... args) const {
    return this->forward(::std::forward<Args>(args)...);
  }

  // logits: [batch_size, vocab_size]
  SampleOutput forward(const torch::Tensor& logits) const;

  // helper functions
  // probs: [..., vocab_size]
  static torch::Tensor greedy_sample(const torch::Tensor& probs);

  // probs: [..., vocab_size]
  static torch::Tensor random_sample(const torch::Tensor& probs);

 private:
  // sample from the distribution
  torch::Tensor sample(const torch::Tensor& probs) const;

  std::vector<bool> do_sample_;
  bool all_random_sample_ = true;
  bool all_greedy_sample_ = true;

  // apply top_p then top_k
  torch::Tensor top_p_;
  torch::Tensor top_k_;
};

}  // namespace llm
