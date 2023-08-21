#pragma once

#include <torch/torch.h>
#include <vector>
#include "input_parameters.h"
#include "memory/kv_cache.h"

namespace llm {

class ICausalLM : public torch::nn::Module {
 public:
  ~ICausalLM() override = default;

  virtual torch::Tensor forward(const torch::Tensor& tokens,     // [num_tokens]
                                const torch::Tensor& positions,  // [num_tokens]
                                std::vector<KVCache>& kv_caches,
                                const InputParameters& parameters) const = 0;
};

template <typename Model, typename Sampler>
class CausalLM : public ICausalLM {
 public:
  CausalLM(Model model) : model_(std::move(model)) {}

  torch::Tensor forward(const torch::Tensor& tokens,     // [num_tokens]
                        const torch::Tensor& positions,  // [num_tokens]
                        std::vector<KVCache>& kv_caches,
                        const InputParameters& parameters) const override {
    // get the logits for the next token
    auto logits = model_.forward(tokens, positions, kv_caches, parameters);
    // select the logits for the last token of each sequence
    auto selected_logits = logits.index_select(0, parameters.sample_idx);
    // call samplers
    auto output_tokens = sampler_.sample(selected_logits, parameters);

    return torch::Tensor();
  }

  // operator() allows us to use the module as a function.
  template <typename... Args>
  torch::Tensor operator()(Args&&... args) {
    return this->forward(::std::forward<Args>(args)...);
  }

 private:
  Model model_;

  Sampler sampler_;
}

}  // namespace llm
