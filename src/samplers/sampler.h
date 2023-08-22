#pragma once
#include <glog/logging.h>
#include <torch/torch.h>
#include <torch/types.h>

#include <functional>

namespace llm {

class Sampler final {
 public:
  Sampler(const std::vector<bool>& do_sample,
          const std::vector<int64_t>& seeds,
          const torch::Device& device = torch::kCPU);

  torch::Tensor sample(const torch::Tensor& logits) const;

 private:
  using SampleFunc = std::function<torch::Tensor(const torch::Tensor&)>;
  std::vector<int64_t> seeds_;
  std::vector<SampleFunc> sample_funcs_;
};

}  // namespace llm
