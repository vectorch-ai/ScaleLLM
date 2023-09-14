#include "worker.h"

#include <c10/core/Device.h>
#include <c10/cuda/CUDAGuard.h>
#include <folly/Unit.h>
#include <folly/futures/Future.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <memory>
#include <string>
#include <torch/csrc/distributed/c10d/HashStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <utility>

#include "common/executor.h"
#include "models/parameters.h"
#include "samplers/logits_processor.h"
#include "samplers/sampler.h"
#include "torch_utils/state_dict.h"

namespace llm {

namespace {

std::unique_ptr<c10d::Backend> create_backend(
    int32_t rank,
    int32_t world_size,
    const c10::intrusive_ptr<c10d::Store>& store,
    const torch::Device& device) {
  if (device.is_cuda()) {
    at::cuda::set_device(device.index());
    // using nccl for cuda
    c10::intrusive_ptr<c10d::ProcessGroupNCCL::Options> opts =
        c10::make_intrusive<c10d::ProcessGroupNCCL::Options>();
    // set as high priority stream
    // opts->is_high_priority_stream = true;
    opts->timeout = std::chrono::milliseconds(60 * 1000);
    return std::make_unique<::c10d::ProcessGroupNCCL>(
        store, rank, world_size, std::move(opts));
  }

  LOG(FATAL) << "Only support CUDA device for now.";
  return nullptr;
}
}  // namespace

Worker::Worker(int32_t rank,
               int32_t world_size,
               const torch::ScalarType& dtype,
               const torch::Device& device)
    : dtype_(dtype), device_(device) {
  parallel_args_.rank(rank);
  parallel_args_.world_size(world_size);
}

bool Worker::init_model(const c10::intrusive_ptr<c10d::Store>& store,
                        const ModelArgs& args) {
  // initialize process group if needed
  if (parallel_args_.world_size() > 1) {
    process_group_ = create_backend(
        parallel_args_.rank(), parallel_args_.world_size(), store, device_);

    torch::DeviceGuard device_guard(device_);
    // warm up the process group
    {
      std::vector<at::Tensor> dummy_tensors{torch::ones({10}, device_)};
      process_group_->allreduce(dummy_tensors)->wait();
    }

    {
      auto input = torch::ones({10}, device_);
      std::vector<torch::Tensor> output;
      output.reserve(parallel_args_.world_size());
      for (int i = 0; i < parallel_args_.world_size(); ++i) {
        output.emplace_back(torch::empty_like(input));
      }
      std::vector<at::Tensor> inputs{input};
      std::vector<std::vector<at::Tensor>> outputs{output};
      process_group_->allgather(outputs, inputs)->wait();
    }

    LOG(INFO) << "Process group initialized for rank: " << parallel_args_.rank()
              << " world size: " << parallel_args_.world_size();

    parallel_args_.process_group(process_group_.get());
  }

  // initialize model
  args_ = args;
  model_ = CausalLM::create(args, parallel_args_, dtype_, device_);
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
  CHECK(model_ != nullptr);
  model_->load_state_dict(state_dict);
}

OutputParameters Worker::execute_model(
    torch::Tensor tokens,     // [num_tokens]
    torch::Tensor positions,  // [num_tokens]
    const InputParameters& params,
    const SamplingParameters& sampling_params) {
  torch::DeviceGuard device_guard(device_);

  torch::Device input_device = tokens.device();

  // all tensors should be on the same device as model
  tokens = tokens.to(device_);
  positions = positions.to(device_);
  InputParameters d_params = params.to(device_);

  // call model forward and return the result
  auto logits = model_->forward(tokens, positions, kv_caches_, d_params);

  // select logits for each sequence
  logits = logits.index_select(/*dim=*/0, d_params.sample_idx);

  // create and call logits processors
  auto logits_processor =
      LogitsProcessor::create(sampling_params, dtype_, device_);
  logits = logits_processor->forward(d_params.token_ids, logits);

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
    torch::Tensor tokens,     // [num_tokens]
    torch::Tensor positions,  // [num_tokens]
    const InputParameters& params,
    const SamplingParameters& sampling_params) {
  folly::Promise<OutputParameters> promise;
  auto future = promise.getSemiFuture();
  executor_.schedule([this,
                      tokens = tokens,
                      positions = positions,
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
folly::SemiFuture<bool> Worker::init_model_async(
    const c10::intrusive_ptr<c10d::Store>& store,
    const ModelArgs& args) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  executor_.schedule(
      [this, store, &args, promise = std::move(promise)]() mutable {
        const bool success = this->init_model(store, args);
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
