#pragma once

#include <folly/futures/Future.h>
#include <torch/torch.h>

#include <string>
#include <utility>

#include "executor.h"
#include "torch_utils/state_dict.h"

namespace llm {
// class CacheManager;

struct InputParameters {};

class Worker final {
 public:
  Worker(std::string model_path, uint32_t rank)
      : model_path_(std::move(model_path)) {}

  ~Worker() = default;

  // Load the model from the given path. blocking call
  // can be called multiple times to reload the model with different parameters
  bool load_state_dict(const StateDict& state_dict);

  // Run the model on the given input. async call
  // input contains prefill and generate requests with the following format:
  // tokens: tokens from all input sequences concatenated together to 1D tensor
  //          [s0_0, s0_1, s0_2, s1_*, s2_*, ..., sN_*, | g0, g1, g2, ..., gM]
  // positions: the position of the tokens in the sequence
  //          [0,    1,   2,     0..., 0..., ...,  0...,| 3,  2,   8, ..., 10]
  // slots: the key value cache physical slots for each token, only used for
  // generate requests
  //                                                     [4,  3,   2, ..., 5]
  // input parameters:
  folly::SemiFuture<bool> execute_model_async(
      const torch::Tensor& tokens,     // [num_tokens]
      const torch::Tensor& positions,  // [num_tokens]
      const torch::Tensor& slots,      // [num_tokens] key value cache slots
      const InputParameters& parameters);

  // initialize model, cache manager. async call
  folly::SemiFuture<bool> init_async();

  // Run the model on the given input. blocking call
  void execute_model(
      const torch::Tensor& tokens,     // [num_tokens]
      const torch::Tensor& positions,  // [num_tokens]
      const torch::Tensor& slots,      // [num_tokens] key value cache slots
      const InputParameters& parameters) const;

  // initialize model, cache manager. blocking call
  bool init();

 private:
  // model path
  std::string model_path_;

  // working thread
  Executor executor_;

  // device to run the model on
  torch::Device device_{"cpu"};

  // cache manager
  // std::unique_ptr<CacheManager> cache_manager_;

  // model
  // std::unique_ptr<Transformer> model_;
};

}  // namespace llm
