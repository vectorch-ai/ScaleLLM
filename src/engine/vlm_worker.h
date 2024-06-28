#pragma once

#include <folly/futures/Future.h>
#include <torch/torch.h>

#include "common/threadpool.h"
#include "model_loader/state_dict.h"
#include "model_parallel/parallel_args.h"
#include "model_runner.h"
#include "models/causal_vlm.h"
#include "models/model_args.h"
#include "models/parameters.h"
#include "parameters.h"
#include "quantization/quant_args.h"

namespace llm {

class VLMWorker final {
 public:
  VLMWorker(const ParallelArgs& parallel_args,
            const torch::Device& device,
            const ModelRunner::Options& runner_options);

  ~VLMWorker() = default;

  // initialize model, cache manager. blocking call
  bool init_model(torch::ScalarType dtype,
                  const ModelArgs& args,
                  const QuantArgs& quant_args);

  torch::Tensor vision_encode(torch::Tensor image, torch::Tensor tokens);

  // Load the model weights from state_dict. blocking call
  // can be called multiple times to reload the model with different parameters
  void load_state_dict(const StateDict& state_dict);

  // verify if the model is loaded correctly
  void verify_loaded_weights() const;

  // returns available memory and total memory
  std::tuple<int64_t, int64_t> profile_device_memory();

  // initialize kv cache. blocking call
  bool init_kv_cache(const std::vector<int64_t>& kv_cache_shape);

  // Run the model on the given input. blocking call
  std::optional<ModelOutput> execute_model(const ModelInput& inputs);

  // capture cuda graph for the model. blocking call
  void capture_cuda_graph(uint32_t batch_size);

  // initialize model, cache manager. async call
  folly::SemiFuture<bool> init_model_async(torch::ScalarType dtype,
                                           const ModelArgs& args,
                                           const QuantArgs& quant_args);

  // Load the model weights from state_dict. async call
  // the future returns a successfull status with no meaningful value
  folly::SemiFuture<folly::Unit> load_state_dict_async(
      const StateDict& state_dict);

  folly::SemiFuture<std::tuple<int64_t, int64_t>> profile_device_memory_async();

  // initialize kv cache. async call
  folly::SemiFuture<bool> init_kv_cache_async(
      const std::vector<int64_t>& kv_cache_shape);

  // Run the model on the given input. async call
  // the future returns a successfull status with no meaningful value
  folly::SemiFuture<std::optional<ModelOutput>> execute_model_async(
      const ModelInput& inputs);

  folly::SemiFuture<folly::Unit> process_group_test_async();

  // capture cuda graph for the model. async call
  folly::SemiFuture<folly::Unit> capture_cuda_graph_async(uint32_t batch_size);

  const torch::Device& device() const { return device_; }

 private:
  void process_group_test();

  // whether the worker is a driver, who takes care of the sampling
  bool driver_ = false;

  // working thread
  ThreadPool threadpool_;

  // dtype of the model
  torch::ScalarType dtype_;

  // device to run the model on
  torch::Device device_;

  // parallel args
  ParallelArgs parallel_args_;

  // model args
  ModelArgs args_;

  // kv caches
  std::vector<llm::KVCache> kv_caches_;

  // causal VLM model
  std::unique_ptr<CausalVLM> model_;

  // runner options
  ModelRunner::Options runner_options_;

  // model runner that runs the model, with cuda graph if enabled
  std::unique_ptr<ModelRunner> model_runner_;
};

}  // namespace llm
