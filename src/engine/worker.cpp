#include "worker.h"

#include <ATen/cuda/CUDAGraph.h>
#include <c10/core/Device.h>
#include <c10/cuda/CUDAGuard.h>
#include <folly/Unit.h>
#include <folly/futures/Future.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <memory>
#include <utility>

#include "common/threadpool.h"
#include "engine/utils.h"
#include "memory/kv_cache.h"
#include "memory/memory.h"
#include "model_loader/state_dict.h"
#include "models/parameters.h"
#include "sampling/logits_processor.h"
#include "sampling/sampler.h"

namespace llm {

const static std::vector<int> BatchSizeForCudaGraph = {
    1,   2,   4,   8,   16,  24,  32,  40,  48,  56,  64,  72,
    80,  88,  96,  104, 112, 120, 128, 136, 144, 152, 160, 168,
    176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256};
constexpr int MaxBatchSizeForCudaGraph = 256;
constexpr int64_t max_seq_len = 1024;

class CudaGraphRunner {
 public:
  explicit CudaGraphRunner(CausalLM* model) : model_(model) {}
  ~CudaGraphRunner() = default;

  void capture(torch::Tensor flatten_tokens,
               torch::Tensor flatten_positions,
               const InputParameters& params,
               std::vector<KVCache>& kv_cache) {
    // run model once to avoid captured graph does not include initial
    // benchmarking
    model_->forward(flatten_tokens, flatten_positions, kv_cache, params);
    torch::cuda::synchronize();

    // create cudagraph and capture
    graph_ = std::make_unique<at::cuda::CUDAGraph>();
    // TODO: optimize could share memorypool between different CUDAGraph
    graph_->capture_begin();
    auto hidden_states =
        model_->forward(flatten_tokens, flatten_positions, kv_cache, params);
    graph_->capture_end();
    torch::cuda::synchronize();

    // save buffers
    flatten_tokens_buffer_ = flatten_tokens;
    flatten_positions_buffer_ = flatten_positions;
    hidden_states_buffer_ = hidden_states;
    new_cache_slots_buffer_ = params.new_cache_slots;
    block_tables_buffer_ = params.block_tables;
    q_cu_seq_lens_buffer_ = params.q_cu_seq_lens;
    kv_cu_seq_lens_buffer_ = params.kv_cu_seq_lens;
  }

  torch::Tensor forward(torch::Tensor flatten_tokens,
                        torch::Tensor flatten_positions,
                        const InputParameters& params,
                        std::vector<KVCache>& kv_cache) {
    flatten_tokens_buffer_.copy_(flatten_tokens, false);
    flatten_positions_buffer_.copy_(flatten_positions, false);
    new_cache_slots_buffer_.copy_(params.new_cache_slots, false);
    block_tables_buffer_.copy_(params.block_tables, false);
    q_cu_seq_lens_buffer_.copy_(params.q_cu_seq_lens, false);
    kv_cu_seq_lens_buffer_.copy_(params.kv_cu_seq_lens, false);
    graph_->replay();
    return hidden_states_buffer_;
  }

 private:
  CausalLM* model_;
  std::unique_ptr<at::cuda::CUDAGraph> graph_;
  // inputs buffer
  torch::Tensor flatten_tokens_buffer_;
  torch::Tensor flatten_positions_buffer_;
  torch::Tensor new_cache_slots_buffer_;
  torch::Tensor block_tables_buffer_;
  torch::Tensor q_cu_seq_lens_buffer_;
  torch::Tensor kv_cu_seq_lens_buffer_;
  // outputs buffer
  torch::Tensor hidden_states_buffer_;
};

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

bool Worker::warmup_model(bool enable_cudagraph) {
  if (enable_cudagraph) {
    LOG(INFO) << "CUDAGraph is enabled.";
    capture_graph();
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

ModelOutput Worker::execute_model(const ModelInput& inputs) {
  torch::DeviceGuard device_guard(device_);

  // all tensors should be on the same device as model
  auto flatten_tokens = inputs.token_ids.to(device_);
  auto flatten_positions = inputs.positions.to(device_);
  InputParameters params = inputs.input_params.to(device_);

  // call model forward to get hidden states
  auto hidden_states =
      model_->forward(flatten_tokens, flatten_positions, kv_caches_, params);

  // waits for all kernels in all streams to complete.
  torch::cuda::synchronize();

  // prepare model output
  ModelOutput output;
  if (inputs.sampling_params.selected_token_idxes.defined()) {
    SamplingParameters sampling_params =
        inputs.sampling_params.to(device_, dtype_);
    // call model to get logits
    torch::Tensor logits =
        model_->logits(hidden_states, sampling_params.selected_token_idxes);

    // create and call logits processors
    auto logits_processor = LogitsProcessor::create(sampling_params);
    // apply logits processors to logits (in place)
    logits = logits_processor->forward(logits,
                                       sampling_params.unique_token_ids,
                                       sampling_params.unique_token_counts,
                                       sampling_params.unique_token_ids_lens);
    // set logits to output
    output.logits = logits;

    auto sampler = std::make_unique<Sampler>(sampling_params.do_sample);
    // select sample logits
    auto sample_logits =
        logits.index_select(/*dim=*/0, sampling_params.sample_idxes);
    auto sample_output = sampler->forward(sample_logits);
    // set sample output to output
    output.sample_output = sample_output;

    // carry over the sampling params
    output.do_sample = sampling_params.do_sample;
  }
  return output;
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

folly::SemiFuture<ModelOutput> Worker::execute_model_async(
    const ModelInput& inputs) {
  folly::Promise<ModelOutput> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule(
      [this, inputs = inputs, promise = std::move(promise)]() mutable {
        // run the model on the given input in working thread
        const auto output = this->execute_model(inputs);
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

folly::SemiFuture<bool> Worker::warmup_model_async(bool enable_cudagraph) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule(
      [this, enable_cudagraph, promise = std::move(promise)]() mutable {
        const bool success = this->warmup_model(enable_cudagraph);
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

// Only support decode in CUDAGraph
void Worker::capture_graph() {
  const int64_t max_seq_len = 1024;  // TODO: need to optimize
  for (auto batch_size : BatchSizeForCudaGraph) {
    torch::Tensor flatten_token_ids;
    torch::Tensor flatten_positions;
    InputParameters input_params;
    Utils::prepare_capture_inputs(max_seq_len,
                                  batch_size,
                                  &flatten_token_ids,
                                  &flatten_positions,
                                  &input_params);
    auto graph_runner = new CudaGraphRunner(model_.get());
    graph_runner->capture(flatten_token_ids,
                          flatten_positions,
                          input_params,
                          // TODO: if use shared buffer, we need to share
                          // memory pool between different cudagraph
                          kv_caches_);
    // TODO graph_memory_pool = graph_runner.graph.pool()
    graph_runners_[batch_size] = graph_runner;
  }
}

}  // namespace llm
