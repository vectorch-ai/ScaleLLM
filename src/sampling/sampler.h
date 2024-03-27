#pragma once
#include <c10/core/TensorOptions.h>
#include <torch/torch.h>
#include <torch/types.h>

#include <functional>

#include "sampling/parameters.h"
namespace llm {

class Sampler final {
 public:
  Sampler(const SamplingParameters& params);

  // operator() allows us to use the module as a function.
  template <typename... Args>
  auto operator()(Args&&... args) const {
    return this->forward(::std::forward<Args>(args)...);
  }

  SampleOutput forward(const torch::Tensor& logits) const;

  // helper functions
  static torch::Tensor greedy_sample(const torch::Tensor& probs);

  static torch::Tensor random_sample(const torch::Tensor& probs);

 private:
  // sample from the distribution
  torch::Tensor sample(const torch::Tensor& probs) const;

  using SampleFunc = std::function<torch::Tensor(const torch::Tensor&)>;
  std::vector<int64_t> seeds_;
  std::vector<SampleFunc> sample_funcs_;

  // apply top_p then top_k
  torch::Tensor top_p_;
  torch::Tensor top_k_;
};

}  // namespace llm
