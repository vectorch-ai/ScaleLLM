#include "worker.h"

#include <c10/core/Device.h>
#include <c10/cuda/CUDAGuard.h>
#include <folly/Unit.h>
#include <folly/futures/Future.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <memory>
#include <utility>

#include "common/threadpool.h"
#include "memory/kv_cache.h"
#include "memory/memory.h"
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
  const auto options = torch::dtype(dtype_).device(device_);
  model_ = CausalLM::create(args, quant_args, parallel_args_, options);
  CHECK(model_ != nullptr) << "Failed to create model.";
  return true;
}

bool Worker::init_kv_cache(const std::vector<int64_t>& kv_cache_shape) {
  CHECK(model_ != nullptr) << "Model is not initialized.";
  // create a KVCache for each layer
  const int64_t num_layers = args_.n_layers();
  kv_caches_.reserve(num_layers);
  for (int64_t i = 0; i < num_layers; ++i) {
    auto key_cache =
        torch::empty(kv_cache_shape, torch::dtype(dtype_).device(device_));
    auto value_cache =
        torch::empty(kv_cache_shape, torch::dtype(dtype_).device(device_));
    kv_caches_.emplace_back(key_cache, value_cache);
  }
  return true;
}

void Worker::load_state_dict(const StateDict& state_dict) {
  CHECK(model_ != nullptr) << "Model is not initialized.";
  model_->load_state_dict(state_dict);
}

void Worker::verify_loaded_weights() const {
  CHECK(model_ != nullptr) << "Model is not initialized.";
  model_->verify_loaded_weights();
}

std::tuple<int64_t, int64_t> Worker::profile_device_memory(
    torch::Tensor flatten_tokens,     // [num_tokens]
    torch::Tensor flatten_positions,  // [num_tokens]
    const InputParameters& params) {
  CHECK(model_ != nullptr) << "Model is not initialized.";
  CHECK(device_.is_cuda()) << "Memory profiling is only supported on GPU.";

  torch::DeviceGuard device_guard(device_);

  // initialize dummy kv caches for profiling
  std::vector<KVCache> dummy_kv_caches(args_.n_layers());

  // release all unocupied cached memory
  // torch::cuda::empty_cache();
  c10::cuda::CUDACachingAllocator::emptyCache();

  // call model forward and discard the result
  model_->forward(flatten_tokens.to(device_),
                  flatten_positions.to(device_),
                  dummy_kv_caches,
                  params.to(device_));

  // waits for all kernels in all streams to complete.
  torch::cuda::synchronize();

  const auto available_memory = memory::available_memory(device_);
  const auto total_memory = memory::total_memory(device_);

  return {available_memory, total_memory};
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

  // call model forward to get hidden states
  auto hidden_states =
      model_->forward(flatten_tokens, flatten_positions, kv_caches_, d_params);

  // call model logits to get logits
  auto logits = model_->logits(hidden_states,
                               sampling_params.last_token_idxes.to(device_));

  // waits for all kernels in all streams to complete.
  torch::cuda::synchronize();

  // TODO: use a seperate stream for sampling parameters tensor copy

  // create and call logits processors
  const auto options = torch::dtype(dtype_).device(device_);
  auto logits_processor = LogitsProcessor::create(sampling_params, options);
  // apply logits processors to logits in-place
  logits_processor->forward(logits,
                            sampling_params.token_ids.to(device_),
                            sampling_params.token_counts.to(device_),
                            sampling_params.token_ids_lens.to(device_));

  // create and call sampler
  auto sampler = std::make_unique<Sampler>(sampling_params, options);
  auto next_tokens = sampler->forward(logits);

  // prepare output parameters
  OutputParameters output_params;
  output_params.next_tokens = next_tokens.to(input_device);
  return output_params;
}

OutputParameters Worker::validate(torch::Tensor flatten_tokens,
                                  torch::Tensor flatten_positions,
                                  const InputParameters& params,
                                  const SamplingParameters& sampling_params) {
  torch::DeviceGuard device_guard(device_);

  torch::Device input_device = flatten_tokens.device();

  flatten_tokens = flatten_tokens.to(device_);
  flatten_positions = flatten_positions.to(device_);
  InputParameters d_params = params.to(device_);

  // call model forward to get hidden states
  auto hidden_states =
      model_->forward(flatten_tokens, flatten_positions, kv_caches_, d_params);

  // call model logits to get logits
  auto logits = model_->logits(hidden_states,
                               sampling_params.last_token_idxes.to(device_));

  const auto options = torch::dtype(dtype_).device(device_);
  auto logits_processor = LogitsProcessor::create(sampling_params, options);

  // TODO: need to support validate multiple speculative steps
  logits_processor->forward(logits,
                            sampling_params.token_ids.to(device_),
                            sampling_params.token_counts.to(device_),
                            sampling_params.token_ids_lens.to(device_));

  auto sampler = std::make_unique<Sampler>(sampling_params, options);
  auto next_tokens = sampler->forward(logits);

  OutputParameters output_params;
  output_params.next_tokens = next_tokens.to(input_device);
  return output_params;
}

folly::SemiFuture<std::tuple<int64_t, int64_t>>
Worker::profile_device_memory_async(
    torch::Tensor flatten_tokens,     // [num_tokens]
    torch::Tensor flatten_positions,  // [num_tokens]
    const InputParameters& params) {
  folly::Promise<std::tuple<int64_t, int64_t>> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this,
                        tokens = flatten_tokens,
                        positions = flatten_positions,
                        parameters = params,
                        promise = std::move(promise)]() mutable {
    const auto output =
        this->profile_device_memory(tokens, positions, parameters);
    promise.setValue(output);
  });
  return future;
}

folly::SemiFuture<OutputParameters> Worker::execute_model_async(
    torch::Tensor flatten_tokens,     // [num_tokens]
    torch::Tensor flatten_positions,  // [num_tokens]
    const InputParameters& params,
    const SamplingParameters& sampling_params) {
  folly::Promise<OutputParameters> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this,
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

folly::SemiFuture<OutputParameters> Worker::validate_async(
    torch::Tensor flatten_tokens,
    torch::Tensor flatten_positions,
    const InputParameters& params,
    const SamplingParameters& sampling_params) {
  folly::Promise<OutputParameters> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this,
                        tokens = flatten_tokens,
                        positions = flatten_positions,
                        parameters = params,
                        sampling_params = sampling_params,
                        promise = std::move(promise)]() mutable {
    const auto output =
        this->validate(tokens, positions, parameters, sampling_params);
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
  threadpool_.schedule([this,
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
    const std::vector<int64_t>& kv_cache_shape) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule(
      [this, &kv_cache_shape, promise = std::move(promise)]() mutable {
        const bool success = this->init_kv_cache(kv_cache_shape);
        promise.setValue(success);
      });
  return future;
}

folly::SemiFuture<folly::Unit> Worker::load_state_dict_async(
    const StateDict& state_dict) {
  folly::Promise<folly::Unit> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule(
      [this, &state_dict, promise = std::move(promise)]() mutable {
        // load the model weights from state_dict within the working thread
        this->load_state_dict(state_dict);
        promise.setValue();
      });
  return future;
}

}  // namespace llm
