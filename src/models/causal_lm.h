#pragma once

#include <c10/core/Device.h>
#include <torch/torch.h>

#include <vector>

#include "memory/kv_cache.h"
#include "model_args.h"
#include "model_loader/state_dict.h"
#include "model_parallel/parallel_args.h"
#include "parameters.h"
#include "quantization/quant_args.h"

namespace llm {

// An interface for causal language models that can hold different models.
class CausalLM {
 public:
  virtual ~CausalLM() = default;

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  // returns: [num_tokens, hidden_size]
  virtual torch::Tensor forward(const torch::Tensor& tokens,
                                const torch::Tensor& positions,
                                std::vector<KVCache>& kv_caches,
                                const InputParameters& parameters) = 0;

  // hidden_states: [num_tokens, hidden_size]
  // seleted_idxes: [num_tokens]
  // returns: [num_tokens, vocab_size]
  virtual torch::Tensor logits(const torch::Tensor& hidden_states,
                               const torch::Tensor& seleted_idxes) = 0;

  // load the model from the given state_dict
  virtual void load_state_dict(const StateDict& state_dict) = 0;

  // verify if the model is loaded correctly
  virtual void verify_loaded_weights() const = 0;

  virtual torch::Device device() const = 0;

  virtual const torch::TensorOptions& options() const = 0;

  // factory method to create a causal language model
  static std::unique_ptr<CausalLM> create(const ModelArgs& args,
                                          const QuantArgs& quant_args,
                                          const ParallelArgs& parallel_args,
                                          const torch::TensorOptions& options);
};

// an template class to hold different models without using virtual functions.
template <typename Model>
class CausalLMImpl : public CausalLM {
 public:
  CausalLMImpl(Model model, const torch::TensorOptions& options)
      : model_(std::move(model)), options_(options) {}

  torch::Tensor forward(const torch::Tensor& tokens,     // [num_tokens]
                        const torch::Tensor& positions,  // [num_tokens]
                        std::vector<KVCache>& kv_caches,
                        const InputParameters& parameters) override {
    return model_->forward(tokens, positions, kv_caches, parameters);
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) override {
    return model_->logits(hidden_states, seleted_idxes);
  }

  void load_state_dict(const StateDict& state_dict) override {
    model_->load_state_dict(state_dict);
  }

  void verify_loaded_weights() const override {
    return model_->verify_loaded_weights();
  }

  torch::Device device() const override { return options_.device(); }

  const torch::TensorOptions& options() const override { return options_; }

 private:
  // underlying model
  Model model_;

  // tensor options
  torch::TensorOptions options_;
};

}  // namespace llm
