#include "worker.h"

#include <c10/core/Device.h>
#include <c10/cuda/CUDAGuard.h>
#include <folly/Unit.h>
#include <folly/futures/Future.h>
#include <torch/torch.h>

#include <memory>
#include <utility>

#include "common/executor.h"
#include "common/logging.h"
#include "model_loader/state_dict.h"
#include "models/input_parameters.h"
#include "sampling/logits_processor.h"
#include "sampling/sampler.h"

namespace llm {

Worker::Worker(const ParallelArgs& parallel_args, const torch::Device& device)
    : parallel_args_(parallel_args), device_(device) {}

bool Worker::init_model(torch::ScalarType dtype,
                        const ModelArgs& args,
                        const QuantArgs& quant_args) {
  // initialize model
  args_ = args;
  dtype_ = dtype;
  model_ = CausalLM::create(args, quant_args, parallel_args_, dtype_, device_);
  GCHECK(model_ != nullptr) << "Failed to create model.";
  return true;
}

bool Worker::init_kv_cache(const std::vector<int64_t>& key_cache_shape,
                           const std::vector<int64_t>& value_cache_shape) {
  // create a KVCache for each layer
  const int64_t num_layers = args_.n_layers();
  kv_caches_.reserve(num_layers);
  for (int64_t i = 0; i < num_layers; ++i) {
    auto key_cache =
        torch::empty(key_cache_shape, torch::dtype(dtype_).device(device_));
    auto value_cache =
        torch::empty(value_cache_shape, torch::dtype(dtype_).device(device_));
    kv_caches_.emplace_back(key_cache, value_cache);
  }
  return true;
}

void Worker::load_state_dict(const StateDict& state_dict) {
  GCHECK(model_ != nullptr);
  model_->load_state_dict(state_dict);
}

void Worker::verify_loaded_weights() const {
  GCHECK(model_ != nullptr);
  model_->verify_loaded_weights();
}

OutputParameters Worker::execute_model(
    torch::Tensor flatten_tokens,     // [num_tokens]
    torch::Tensor flatten_positions,  // [num_tokens]
    const InputParameters& params,
    const SamplingParameters& sampling_params) {
  torch::DeviceGuard device_guard(device_);

  torch::Device input_device = flatten_tokens.device();

  // all tensors should be on the same device as model
  flatten_tokens = flatten_tokens.to(device_);
  flatten_positions = flatten_positions.to(device_);
  InputParameters d_params = params.to(device_);

  // call model forward and return the result
  auto logits =
      model_->forward(flatten_tokens, flatten_positions, kv_caches_, d_params);

  // create and call logits processors
  auto logits_processor =
      LogitsProcessor::create(sampling_params, dtype_, device_);
  logits = logits_processor->forward(d_params.token_ids,
                                     d_params.token_counts,
                                     d_params.token_ids_lens,
                                     logits);

  // create and call sampler
  auto sampler = std::make_unique<Sampler>(
      sampling_params.do_sample, sampling_params.seeds, device_);
  auto next_tokens = sampler->sample(logits);

  // prepare output parameters
  OutputParameters output_params;
  output_params.next_tokens = next_tokens.to(input_device);
  return output_params;
}

folly::SemiFuture<OutputParameters> Worker::execute_model_async(
    torch::Tensor flatten_tokens,     // [num_tokens]
    torch::Tensor flatten_positions,  // [num_tokens]
    const InputParameters& params,
    const SamplingParameters& sampling_params) {
  folly::Promise<OutputParameters> promise;
  auto future = promise.getSemiFuture();
  executor_.schedule([this,
                      tokens = flatten_tokens,
                      positions = flatten_positions,
                      parameters = params,
                      sampling_params = sampling_params,
                      promise = std::move(promise)]() mutable {
    // run the model on the given input in working thread
    const auto output =
        this->execute_model(tokens, positions, parameters, sampling_params);
    promise.setValue(output);
  });
  return future;
}

// initialize model, cache manager. async call
folly::SemiFuture<bool> Worker::init_model_async(torch::ScalarType dtype,
                                                 const ModelArgs& args,
                                                 const QuantArgs& quant_args) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  executor_.schedule([this,
                      dtype,
                      &args,
                      &quant_args,
                      promise = std::move(promise)]() mutable {
    const bool success = this->init_model(dtype, args, quant_args);
    promise.setValue(success);
  });
  return future;
}

folly::SemiFuture<bool> Worker::init_kv_cache_async(
    const std::vector<int64_t>& key_cache_shape,
    const std::vector<int64_t>& value_cache_shape) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  executor_.schedule([this,
                      &key_cache_shape,
                      &value_cache_shape,
                      promise = std::move(promise)]() mutable {
    const bool success =
        this->init_kv_cache(key_cache_shape, value_cache_shape);
    promise.setValue(success);
  });
  return future;
}

folly::SemiFuture<folly::Unit> Worker::load_state_dict_async(
    const StateDict& state_dict) {
  folly::Promise<folly::Unit> promise;
  auto future = promise.getSemiFuture();
  executor_.schedule(
      [this, &state_dict, promise = std::move(promise)]() mutable {
        // load the model weights from state_dict within the working thread
        this->load_state_dict(state_dict);
        promise.setValue();
      });
  return future;
}

}  // namespace llm
