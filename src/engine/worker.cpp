#include "worker.h"

#include <ATen/cuda/CUDAGraph.h>
#include <c10/core/Device.h>
#include <c10/cuda/CUDAGuard.h>
#include <folly/Unit.h>
#include <folly/futures/Future.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <memory>
#include <optional>
#include <utility>

#include "common/metrics.h"
#include "common/threadpool.h"
#include "common/timer.h"
#include "memory/kv_cache.h"
#include "memory/memory.h"
#include "model_loader/state_dict.h"
#include "model_parallel/model_parallel.h"
#include "models/parameters.h"
#include "sampling/logits_processor.h"
#include "sampling/sampler.h"

// latency metrics
DEFINE_COUNTER_FAMILY(execution_latency_seconds,
                      "Execution latency in seconds");
DEFINE_COUNTER_INSTANCE(model_execution_latency_seconds,
                        execution_latency_seconds,
                        {{"stage", "model"}});
DEFINE_COUNTER_INSTANCE(logits_processing_latency_seconds,
                        execution_latency_seconds,
                        {{"stage", "logits_processing"}});
DEFINE_COUNTER_INSTANCE(sampling_latency_seconds,
                        execution_latency_seconds,
                        {{"stage", "sampling"}});

namespace llm {

Worker::Worker(const ParallelArgs& parallel_args,
               const torch::Device& device,
               const ModelRunner::Options& runner_options)
    : parallel_args_(parallel_args),
      device_(device),
      runner_options_(runner_options) {
  // first worker is the driver
  driver_ = parallel_args.rank() == 0;
}

bool Worker::init_model(torch::ScalarType dtype,
                        const ModelArgs& args,
                        const QuantArgs& quant_args) {
  CHECK(model_ == nullptr) << "Model is already initialized.";

  // initialize model
  args_ = args;
  dtype_ = dtype;
  const auto options = torch::dtype(dtype_).device(device_);
  model_ = CausalLM::create(args, quant_args, parallel_args_, options);
  CHECK(model_ != nullptr) << "Failed to create model.";
  model_runner_ =
      std::make_unique<ModelRunner>(model_.get(), device_, runner_options_);
  return true;
}

bool Worker::init_kv_cache(const std::vector<int64_t>& kv_cache_shape) {
  CHECK(model_ != nullptr) << "Model is not initialized.";
  CHECK(kv_caches_.empty()) << "KV caches are already initialized.";

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

bool Worker::capture_cuda_graphs() {
  CHECK(model_ != nullptr) << "Model is not initialized.";
  CHECK(!kv_caches_.empty()) << "KV caches are not initialized.";
  // capture graphs if needed
  model_runner_->capture_cuda_graphs(kv_caches_);
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

std::tuple<int64_t, int64_t> Worker::profile_device_memory() {
  CHECK(model_ != nullptr) << "Model is not initialized.";
  CHECK(device_.is_cuda()) << "Memory profiling is only supported on GPU.";

  const auto available_memory = memory::available_memory(device_);
  const auto total_memory = memory::total_memory(device_);

  return {available_memory, total_memory};
}

bool Worker::process_group_test() {
  torch::DeviceGuard device_guard(device_);
  torch::cuda::synchronize();

  // create random tensors
  const auto options = torch::dtype(torch::kHalf).device(device_);
  torch::Tensor tensor = torch::randn({10, 10}, options);
  // call allreduce
  reduce_from_model_parallel_region(tensor, parallel_args_);
  // call allgather
  gather_from_model_parallel_region(tensor, parallel_args_);

  torch::cuda::synchronize();
  return true;
}

std::optional<ModelOutput> Worker::execute_model(const ModelInput& inputs) {
  torch::DeviceGuard device_guard(device_);
  torch::cuda::synchronize();

  Timer timer;

  // all tensors should be on the same device as model
  auto flatten_tokens = inputs.token_ids.to(device_);
  auto flatten_positions = inputs.positions.to(device_);
  auto params = inputs.input_params.to(device_);
  auto sampling_params = inputs.sampling_params.to(device_, dtype_);

  // call model runner forward to get hidden states
  auto hidden_states = model_runner_->forward(
      flatten_tokens, flatten_positions, kv_caches_, params);

  torch::Tensor logits;
  if (sampling_params.selected_token_idxes.defined()) {
    logits =
        model_->logits(hidden_states, sampling_params.selected_token_idxes);
  }

  torch::cuda::synchronize();
  COUNTER_ADD(model_execution_latency_seconds, timer.elapsed_seconds());

  if (!driver_) {
    return std::nullopt;
  }

  // driver prepare model output
  ModelOutput output;
  if (sampling_params.selected_token_idxes.defined()) {
    // create and call logits processors
    timer.reset();
    auto logits_processor = LogitsProcessor::create(sampling_params);
    // apply logits processors to logits (in place)
    logits = logits_processor->forward(logits,
                                       sampling_params.unique_token_ids,
                                       sampling_params.unique_token_counts,
                                       sampling_params.unique_token_ids_lens);
    COUNTER_ADD(logits_processing_latency_seconds, timer.elapsed_seconds());

    // set logits to output
    output.logits = logits;

    timer.reset();
    auto sampler = std::make_unique<Sampler>(sampling_params.do_sample,
                                             sampling_params.logprobs,
                                             sampling_params.max_top_logprobs);
    // select sample logits
    auto sample_logits =
        logits.index_select(/*dim=*/0, sampling_params.sample_idxes);
    auto sample_output = sampler->forward(sample_logits);
    COUNTER_ADD(sampling_latency_seconds, timer.elapsed_seconds());

    // set sample output to output
    output.sample_output = sample_output;

    // carry over the sampling params
    output.do_sample = sampling_params.do_sample;
    output.logprobs = sampling_params.logprobs;
    output.max_top_logprobs = sampling_params.max_top_logprobs;
  }
  return output;
}

folly::SemiFuture<std::tuple<int64_t, int64_t>>
Worker::profile_device_memory_async() {
  folly::Promise<std::tuple<int64_t, int64_t>> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this, promise = std::move(promise)]() mutable {
    const auto output = this->profile_device_memory();
    promise.setValue(output);
  });
  return future;
}

folly::SemiFuture<std::optional<ModelOutput>> Worker::execute_model_async(
    const ModelInput& inputs) {
  folly::Promise<std::optional<ModelOutput>> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule(
      [this, inputs = inputs, promise = std::move(promise)]() mutable {
        // run the model on the given input in working thread
        const auto output = this->execute_model(inputs);
        promise.setValue(output);
      });
  return future;
}

folly::SemiFuture<bool> Worker::process_group_test_async() {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this, promise = std::move(promise)]() mutable {
    promise.setValue(this->process_group_test());
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

folly::SemiFuture<bool> Worker::capture_cuda_graphs_async() {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this, promise = std::move(promise)]() mutable {
    const bool success = this->capture_cuda_graphs();
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
