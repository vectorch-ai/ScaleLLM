#pragma once
#include <torch/torch.h>
#include <torch/types.h>

#include <functional>

#include "request/sampling_parameter.h"
namespace llm {

class Sampler final {
 public:
  Sampler(const SamplingParameters& params,
          torch::ScalarType dtype,
          const torch::Device& device);

  torch::Tensor forward(const torch::Tensor& logits) const;

  // operator() allows us to use the module as a function.
  template <typename... Args>
  torch::Tensor operator()(Args&&... args) const {
    return this->forward(::std::forward<Args>(args)...);
  }

 private:
  torch::Tensor sample(const torch::Tensor& probs) const;

  using SampleFunc = std::function<torch::Tensor(const torch::Tensor&)>;
  std::vector<int64_t> seeds_;
  std::vector<SampleFunc> sample_funcs_;

  // apply top_p then top_k
  torch::Tensor top_p_;
  torch::Tensor top_k_;
};

}  // namespace llm
