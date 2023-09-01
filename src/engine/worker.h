#pragma once

#include <folly/futures/Future.h>
#include <torch/torch.h>

#include <string>
#include <utility>

#include "executor.h"
#include "models/causal_lm.h"
#include "models/input_parameters.h"
#include "torch_utils/state_dict.h"

namespace llm {
// class CacheManager;

class Worker final {
 public:
  Worker(const torch::Device& device) : device_(device) {}

  ~Worker() = default;

  // initialize model, cache manager. blocking call
  bool init(const ModelArgs& args);

  // Load the model weights from state_dict. blocking call
  // can be called multiple times to reload the model with different parameters
  void load_state_dict(const StateDict& state_dict);

  // Run the model on the given input. blocking call
  OutputParameters execute_model(torch::Tensor tokens,     // [num_tokens]
                                 torch::Tensor positions,  // [num_tokens]
                                 const InputParameters& params,
                                 const SamplingParameters& sampling_params);

  // initialize model, cache manager. async call
  folly::SemiFuture<bool> init_async(const ModelArgs& args);

  // Load the model weights from state_dict. async call
  // the future returns a successfull status with no meaningful value
  folly::SemiFuture<folly::Unit> load_state_dict_async(
      const StateDict& state_dict);

  // Run the model on the given input. async call
  // the future returns a successfull status with no meaningful value
  folly::SemiFuture<OutputParameters> execute_model_async(
      torch::Tensor tokens,     // [num_tokens]
      torch::Tensor positions,  // [num_tokens]
      const InputParameters& params,
      const SamplingParameters& sampling_params);

 private:
  // working thread
  Executor executor_;

  // device to run the model on
  torch::Device device_;

  // kv caches
  std::vector<llm::KVCache> kv_caches_;

  // model
  std::unique_ptr<CausalLM> model_;
};

}  // namespace llm
