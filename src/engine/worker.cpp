#include "worker.h"

#include <c10/core/Device.h>
#include <c10/cuda/CUDAGuard.h>
#include <folly/Unit.h>
#include <folly/futures/Future.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <memory>
#include <string>
#include <utility>

#include "executor.h"
#include "models/parameters.h"
#include "samplers/logits_processor.h"
#include "samplers/sampler.h"
#include "torch_utils/state_dict.h"

namespace llm {

bool Worker::init(const ModelArgs& args) {
  // initialize model
  model_ = CausalLM::create(args, device_);

  // initialize kv caches
  // calculate cache shapes
  const auto element_size = torch::tensor({}).element_size();
  const auto x = 16 / element_size;
  const int64_t block_size = 8;  // 8 slots per block
  const int64_t num_heads = args.n_heads();
  const int64_t head_dim = args.dim() / num_heads;
  const int64_t num_blocks = args.max_seq_len() / block_size + 1;
  const std::vector<int64_t> key_cache_shape = {
      num_blocks, num_heads, static_cast<int64_t>(head_dim / x), block_size, x};
  const std::vector<int64_t> value_cache_shape = {
      num_blocks, num_heads, head_dim, block_size};

  // create a KVCache for each layer
  kv_caches_.reserve(args.n_layers());
  for (int i = 0; i < args.n_layers(); ++i) {
    auto key_cache = torch::zeros(key_cache_shape, device_);
    auto value_cache = torch::zeros(value_cache_shape, device_);
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
  auto logits_processor = LogitsProcessor::create(sampling_params, device_);
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
folly::SemiFuture<bool> Worker::init_async(const ModelArgs& args) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  executor_.schedule(
      [this, &args = args, promise = std::move(promise)]() mutable {
        const bool success = this->init(args);
        promise.setValue(success);
      });
  return future;
}

folly::SemiFuture<folly::Unit> Worker::load_state_dict_async(
    const StateDict& state_dict) {
  folly::Promise<folly::Unit> promise;
  auto future = promise.getSemiFuture();
  executor_.schedule(
      [this, &state_dict = state_dict, promise = std::move(promise)]() mutable {
        // load the model weights from state_dict within the working thread
        this->load_state_dict(state_dict);
        promise.setValue();
      });
  return future;
}

}  // namespace llm
