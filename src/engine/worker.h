#pragma once

#include <folly/futures/Future.h>
#include <torch/torch.h>

#include "common/threadpool.h"
#include "model_loader/state_dict.h"
#include "model_parallel/parallel_args.h"
#include "models/causal_lm.h"
#include "models/input_parameters.h"
#include "models/model_args.h"
#include "quantization/quant_args.h"

namespace llm {

// output parameters for the model that encapsulates all the necessary
// output information. The output parameters should be as small as possible
// to avoid transferring large tensors between host and device.
struct OutputParameters {
  // rerang the next tokens and next logprob according to the seq_idx
  void index_select(const torch::Tensor& seq_idx) {
    next_tokens = next_tokens.index_select(/*dim=*/0, seq_idx);
    // next_logprob = next_logprob.index_select(/*dim=*/0, seq_idx);
  }
  // [num_seq] LongTensor
  torch::Tensor next_tokens;

  // [num_seq]
  // torch::Tensor next_logprob;
};

class CudaGraphRunner;
class Worker final {
 public:
  Worker(const ParallelArgs& parallel_args, const torch::Device& device);

  ~Worker() = default;

  // initialize model, cache manager. blocking call
  bool init_model(torch::ScalarType dtype,
                  const ModelArgs& args,
                  const QuantArgs& quant_args);

  // Load the model weights from state_dict. blocking call
  // can be called multiple times to reload the model with different parameters
  void load_state_dict(const StateDict& state_dict);

  // verify if the model is loaded correctly
  void verify_loaded_weights() const;

  // returns available memory and total memory
  std::tuple<int64_t, int64_t> profile_device_memory(
      torch::Tensor flatten_tokens,     // [num_tokens]
      torch::Tensor flatten_positions,  // [num_tokens]
      const InputParameters& params);

  // initialize kv cache. blocking call
  bool init_kv_cache(const std::vector<int64_t>& kv_cache_shape);

  // Run the model on the given input. blocking call
  OutputParameters execute_model(
      torch::Tensor flatten_tokens,     // [num_tokens]
      torch::Tensor flatten_positions,  // [num_tokens]
      const InputParameters& params,
      const SamplingParameters& sampling_params);

  bool warmup_model(bool enable_cudagraph);

  // TODO
  OutputParameters validate(torch::Tensor flatten_tokens,
                            torch::Tensor flatten_positions,
                            const InputParameters& params,
                            const SamplingParameters& sampling_params);

  // initialize model, cache manager. async call
  folly::SemiFuture<bool> init_model_async(torch::ScalarType dtype,
                                           const ModelArgs& args,
                                           const QuantArgs& quant_args);

  // Load the model weights from state_dict. async call
  // the future returns a successfull status with no meaningful value
  folly::SemiFuture<folly::Unit> load_state_dict_async(
      const StateDict& state_dict);

  folly::SemiFuture<std::tuple<int64_t, int64_t>> profile_device_memory_async(
      torch::Tensor flatten_tokens,     // [num_tokens]
      torch::Tensor flatten_positions,  // [num_tokens]
      const InputParameters& params);

  // initialize kv cache. async call
  folly::SemiFuture<bool> init_kv_cache_async(
      const std::vector<int64_t>& kv_cache_shape);

  // Run the model on the given input. async call
  // the future returns a successfull status with no meaningful value
  folly::SemiFuture<OutputParameters> execute_model_async(
      torch::Tensor flatten_tokens,     // [num_tokens]
      torch::Tensor flatten_positions,  // [num_tokens]
      const InputParameters& params,
      const SamplingParameters& sampling_params);

  folly::SemiFuture<OutputParameters> validate_async(
      torch::Tensor flatten_tokens,
      torch::Tensor flatten_positions,
      const InputParameters& params,
      const SamplingParameters& sampling_params);

  folly::SemiFuture<bool> warmup_model_async(
      bool enable_cudagraph);

  const torch::Device& device() const { return device_; }

 private:
  // capture cuda graph
  void capture_graph();

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

  // model
  std::unique_ptr<CausalLM> model_;

  // graph runner
  std::map<int64_t, CudaGraphRunner*> graph_runners_;
};

}  // namespace llm
