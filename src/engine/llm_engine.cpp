#include "llm_engine.h"

#include <ATen/cuda/CUDAContext.h>
#include <gflags/gflags_declare.h>
#include <glog/logging.h>

#include <boost/algorithm/string.hpp>
#include <memory>

#include "common/pretty_print.h"
#include "model_loader/model_loader.h"
#include "model_parallel/parallel_args.h"
#include "models/model_args.h"
#include "utils.h"
#include "worker.h"

static constexpr int64_t GB = int64_t(1024) * 1024 * 1024;

DEFINE_int32(block_size, 16, "slots per block, value must be multiple of 16");
DEFINE_int64(max_cache_size, 10 * GB, "max cache size in bytes, default 10GB");
DEFINE_double(max_memory_utilization,
              0.9,
              "maximum memory utilization allowed, default 0.9");
DEFINE_bool(enable_cudagraph,
            false,
            "Enable CUDAGraph to optimize model execution.");

// following two parameters are used for profiling and warmup the engine.
// the profiling result would be used to determine kv cache size.
DEFINE_int64(max_num_tokens_per_batch,
             1024,
             "Maximum number of tokens per batch for profiling.");
DEFINE_int64(max_num_seqs_per_batch,
             32,
             "Maximum number of sequences per batch for profiling.");

DECLARE_bool(disable_custom_kernels);

namespace llm {
namespace {
torch::ScalarType parse_dtype(const std::string& dtype_str,
                              const torch::Device& device) {
  if (device.is_cpu()) {
    return torch::kFloat32;
  }

  if (boost::iequals(dtype_str, "half") ||
      boost::iequals(dtype_str, "float16")) {
    return torch::kHalf;
  }
  if (boost::iequals(dtype_str, "bfloat16")) {
    return torch::kBFloat16;
  }
  if ((boost::iequals(dtype_str, "float") ||
       boost::iequals(dtype_str, "float32"))) {
    return torch::kFloat32;
  }

  if (dtype_str.empty() || boost::iequals(dtype_str, "auto")) {
    return torch::kFloat16;
  }
  CHECK(false) << "Unsupported dtype: " << dtype_str << " on device " << device;
}
}  // namespace

LLMEngine::LLMEngine(const std::vector<torch::Device>& devices)
    : devices_(devices) {
  CHECK_GT(devices.size(), 0) << "At least one device is required";

  const auto device_type = devices[0].type();
  for (const auto device : devices) {
    CHECK_EQ(device.type(), device_type)
        << "All devices should be the same type";

    if (device.is_cuda()) {
      // check cuda compute capability
      const auto* properties = at::cuda::getDeviceProperties(device.index());
      const bool is_sm8x = properties->major == 8 && properties->minor >= 0;
      const bool is_sm90 = properties->major == 9 && properties->minor == 0;
      CHECK(is_sm90 || is_sm8x) << "Engine only supports Ampere GPUs or newer.";
      // TODO: add Turing(sm75) support in the near future.
    }
  }

  // initialize process groups if there are multiple devices
  const int32_t world_size = static_cast<int32_t>(devices.size());
  if (world_size > 1) {
    // create a process group for each device if there are multiple gpus
    process_groups_ = ProcessGroup::create_process_groups(devices);
  }

  // create a worker for each device
  for (size_t i = 0; i < devices.size(); ++i) {
    const int32_t rank = static_cast<int32_t>(i);
    ProcessGroup* pg = world_size > 1 ? process_groups_[i].get() : nullptr;
    ParallelArgs parallel_args(rank, world_size, pg);
    workers_.emplace_back(std::make_unique<Worker>(parallel_args, devices[i]));
  }

  if (FLAGS_disable_custom_kernels) {
    LOG(WARNING) << "Custom kernels are disabled. You may experience "
                    "performance degradation.";
  }
}

bool LLMEngine::init(const std::string& model_weights_path) {
  if (!init_model(model_weights_path)) {
    LOG(ERROR) << "Failed to initialize model from: " << model_weights_path;
    return false;
  }

  // initialize kv cache
  const int64_t cache_size_in_bytes = profile_memory_for_kv_cache();
  CHECK_GT(cache_size_in_bytes, 0);
  LOG(INFO) << "Initializing kv cache with size: "
            << readable_size(cache_size_in_bytes);
  const int64_t n_blocks = calculate_kv_cache_blocks(cache_size_in_bytes);
  if (!init_kv_cache(n_blocks)) {
    LOG(ERROR) << "Failed to initialize kv cache";
    return false;
  }
  if (!warmup_model()) {
    LOG(ERROR) << "Failed to warmup model.";
    return false;
  }
  return true;
}

bool LLMEngine::init_model(const std::string& model_weights_path) {
  auto model_loader = ModelLoader::create(model_weights_path);
  LOG(INFO) << "Initializing model from: " << model_weights_path;

  tokenizer_ = model_loader->tokenizer();
  CHECK(tokenizer_ != nullptr);

  args_ = model_loader->model_args();
  quant_args_ = model_loader->quant_args();
  tokenizer_args_ = model_loader->tokenizer_args();

  // compute the number of local kv heads and head dim
  const int world_size = static_cast<int>(workers_.size());
  const int64_t n_heads = args_.n_heads();
  const int64_t n_kv_heads = args_.n_kv_heads().value_or(n_heads);
  n_local_kv_heads_ = n_kv_heads / world_size;
  head_dim_ = args_.hidden_size() / n_heads;
  dtype_ = parse_dtype(args_.dtype(), devices_[0]);

  // key + value for all layers
  LOG(INFO) << "Block info, block_size: " << FLAGS_block_size
            << ", n_local_kv_heads: " << n_local_kv_heads_
            << ", head_dim: " << head_dim_ << ", n_layers: " << args_.n_layers()
            << ", dtype: " << dtype_;

  if (tokenizer_->vocab_size() != args_.vocab_size()) {
    // use tokenizer vocab size if model vocab size is not set
    if (args_.vocab_size() <= 0) {
      LOG(WARNING) << "Model vocab size is not set, using tokenizer vocab "
                      "size: "
                   << tokenizer_->vocab_size();
      args_.vocab_size(tokenizer_->vocab_size());
    } else {
      LOG(WARNING) << "Vocab size mismatch: tokenizer: "
                   << tokenizer_->vocab_size()
                   << ", model: " << args_.vocab_size();
    }
  }

  LOG(INFO) << "Initializing model with " << args_;
  LOG(INFO) << "Initializing model with quant args: " << quant_args_;
  LOG(INFO) << "Initializing model with tokenizer args: " << tokenizer_args_;

  if (workers_.size() == 1) {
    Worker* worker = workers_[0].get();
    // only one worker, call init_model in current thread
    if (!worker->init_model(dtype_, args_, quant_args_)) {
      return false;
    }
    // load the weights from the checkpoint
    for (const auto& state_dict : *model_loader) {
      worker->load_state_dict(state_dict);
    }
    worker->verify_loaded_weights();
    return true;
  }

  // init model for each worker in parallel
  // multiple workers, call async init
  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(workers_.size());
  for (auto& worker : workers_) {
    futures.push_back(worker->init_model_async(dtype_, args_, quant_args_));
  }
  // wait for all futures to complete
  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.value()) {
      return false;
    }
  }

  // load the weights from the checkpoint in parallel
  for (const auto& state_dict : *model_loader) {
    std::vector<folly::SemiFuture<folly::Unit>> futures;
    futures.reserve(workers_.size());
    for (auto& worker : workers_) {
      futures.push_back(worker->load_state_dict_async(state_dict));
    }
    // wait for all futures to complete
    auto results = folly::collectAll(futures).get();
    for (const auto& result : results) {
      if (result.hasException()) {
        return false;
      }
    }
  }

  // verify the weights are loaded correctly
  for (const auto& worker : workers_) {
    worker->verify_loaded_weights();
  }
  return true;
}

bool LLMEngine::warmup_model() {
  if (workers_.size() == 1) {
    // only one worker, call blocking forward
    return workers_[0]->warmup_model(FLAGS_enable_cudagraph);
  }

  // multiple workers, call async forward
  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(workers_.size());
  for (auto& worker : workers_) {
    futures.emplace_back(worker->warmup_model_async(FLAGS_enable_cudagraph));
  }
  // wait for the all future to complete
  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.value()) {
      return false;
    }
  }
  return true;
}

int64_t LLMEngine::profile_memory_for_kv_cache() {
  // use first device to profile memory usage
  const auto& device = workers_[0]->device();
  if (device.is_cpu()) {
    // use max memory cache size for CPU
    LOG(INFO) << "Initializing CPU cache with max cache size: "
              << readable_size(FLAGS_max_cache_size);
    // TODO: add CPU memory profiling
    return FLAGS_max_cache_size;
  }
  CHECK(device.is_cuda()) << "Only support CPU and CUDA device for now.";

  // Prepare dummy inputs for memory profiling
  torch::Tensor flatten_token_ids;
  torch::Tensor flatten_positions;
  InputParameters input_params;
  Utils::prepare_profile_inputs(FLAGS_max_num_tokens_per_batch,
                                FLAGS_max_num_seqs_per_batch,
                                &flatten_token_ids,
                                &flatten_positions,
                                &input_params);
  LOG(INFO) << "Warming up the engine with input shape: "
            << flatten_token_ids.sizes();

  // call worker to profile memory usage
  std::vector<folly::SemiFuture<std::tuple<int64_t, int64_t>>> futures;
  futures.reserve(workers_.size());
  for (auto& worker : workers_) {
    futures.push_back(worker->profile_device_memory_async(
        flatten_token_ids, flatten_positions, input_params));
  }

  // pick smallest available memory from all devices
  int64_t smallest_available_memory = std::numeric_limits<int64_t>::max();
  // wait for all futures to complete
  auto results = folly::collectAll(futures).get();
  for (size_t i = 0; i < results.size(); ++i) {
    const auto device = workers_[i]->device();
    if (!results[i].hasValue()) {
      LOG(ERROR) << "Failed to profile memory usage for device: " << device;
      continue;
    }
    auto [available_memory, total_memory] = results[i].value();
    LOG(INFO) << device
              << ": available memory: " << readable_size(available_memory)
              << ", total memory: " << readable_size(total_memory);

    LOG(INFO) << "Using max_memory_utilization: "
              << FLAGS_max_memory_utilization
              << ", max_cache_size: " << readable_size(FLAGS_max_cache_size);
    // apply memory cap from config if it is set
    if (FLAGS_max_memory_utilization < 1.0) {
      const int64_t buffer_memory =
          total_memory * (1.0 - FLAGS_max_memory_utilization);
      available_memory -= buffer_memory;
    }
    if (FLAGS_max_cache_size > 0) {
      available_memory = std::min(available_memory, FLAGS_max_cache_size);
    }
    smallest_available_memory =
        std::min(smallest_available_memory, available_memory);
  }
  return std::max(smallest_available_memory, int64_t(0));
}

bool LLMEngine::init_kv_cache(int64_t n_blocks) {
  CHECK_GT(n_blocks, 0) << "no memory for kv cache";

  // init kv cache for each worker
  const std::vector<int64_t> kv_cache_shape = {
      n_blocks, FLAGS_block_size, n_local_kv_heads_, head_dim_};
  LOG(INFO) << "Initializing kv cache with shape: [" << kv_cache_shape << "]";

  // initialize block manager
  block_manager_ = std::make_unique<BlockManager>(n_blocks, FLAGS_block_size);

  // init kv cache for each worker in parallel
  if (workers_.size() == 1) {
    // only one worker, call init_kv_cache in current thread
    return workers_[0]->init_kv_cache(kv_cache_shape);
  }

  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(workers_.size());
  for (auto& worker : workers_) {
    futures.push_back(worker->init_kv_cache_async(kv_cache_shape));
  }
  // wait for all futures to complete
  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.value()) {
      return false;
    }
  }
  return true;
}

ModelOutput LLMEngine::execute_model(Batch& batch) {
  // prepare inputs for workers
  auto model_inputs = batch.prepare_model_input();
  if (workers_.size() == 1) {
    // only one worker, call blocking forward
    auto model_output = workers_[0]->execute_model(model_inputs);
    batch.process_sample_output(model_output.sample_output);
    // carry over the sampling params
    model_output.do_sample = model_inputs.sampling_params.do_sample;
    return model_output;
  }

  // multiple workers, call async forward
  std::vector<folly::SemiFuture<ModelOutput>> futures;
  futures.reserve(workers_.size());
  for (auto& worker : workers_) {
    futures.push_back(worker->execute_model_async(model_inputs));
  }
  // wait for the all future to complete
  auto results = folly::collectAll(futures).get();
  // return the result from the first worker
  auto model_output = results.front().value();
  batch.process_sample_output(model_output.sample_output);
  // carry over the sampling params
  model_output.do_sample = model_inputs.sampling_params.do_sample;
  return model_output;
}

int64_t LLMEngine::kv_cache_slot_size_in_bytes() const {
  const auto dtype_size = torch::scalarTypeToTypeMeta(dtype_).itemsize();
  // key + value for all layers
  const int64_t slot_size_in_bytes =
      2 * n_local_kv_heads_ * head_dim_ * args_.n_layers() * dtype_size;
  return slot_size_in_bytes;
}

int64_t LLMEngine::calculate_kv_cache_blocks(
    int64_t cache_size_in_bytes) const {
  const int64_t block_size_in_bytes =
      FLAGS_block_size * kv_cache_slot_size_in_bytes();
  return cache_size_in_bytes / block_size_in_bytes;
}

}  // namespace llm
