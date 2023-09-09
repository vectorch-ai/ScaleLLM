#pragma once

#include <torch/torch.h>

#include <vector>

#include "memory/kv_cache.h"
#include "models/model_args.h"
#include "models/parallel_args.h"
#include "parameters.h"
#include "torch_utils/state_dict.h"

namespace llm {

// An interface for causal language models that can hold different models.
class CausalLM : public torch::nn::Module {
 public:
  ~CausalLM() override = default;

  // returns logits of shape [num_tokens, vocab_size]
  virtual torch::Tensor forward(const torch::Tensor& tokens,     // [num_tokens]
                                const torch::Tensor& positions,  // [num_tokens]
                                std::vector<KVCache>& kv_caches,
                                const InputParameters& parameters) = 0;

  // load the model from the given state_dict
  virtual void load_state_dict(const StateDict& state_dict) = 0;

  // factory method to create a causal language model
  static std::unique_ptr<CausalLM> create(const ModelArgs& args,
                                          const ParallelArgs& parallel_args,
                                          const torch::ScalarType& dtype,
                                          const torch::Device& device);
};

// an template class to hold different models without using virtual functions.
template <typename Model>
class CausalLMImpl : public CausalLM {
 public:
  CausalLMImpl(Model model) : model_(std::move(model)) {}

  torch::Tensor forward(const torch::Tensor& tokens,     // [num_tokens]
                        const torch::Tensor& positions,  // [num_tokens]
                        std::vector<KVCache>& kv_caches,
                        const InputParameters& parameters) override {
    return model_->forward(tokens, positions, kv_caches, parameters);
  }

  void load_state_dict(const StateDict& state_dict) override {
    model_->load_state_dict(state_dict);
  }

 private:
  Model model_;
};

}  // namespace llm
