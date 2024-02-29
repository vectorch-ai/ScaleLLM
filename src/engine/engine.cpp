#include "engine.h"

#include <ATen/cuda/CUDAContext.h>
#include <gflags/gflags_declare.h>
#include <glog/logging.h>

#include <boost/algorithm/string.hpp>
#include <memory>

#include "common/pretty_print.h"
#include "memory/memory.h"
#include "model_loader/model_loader.h"
#include "model_parallel/parallel_args.h"
#include "models/model_args.h"
#include "utils.h"
#include "worker.h"

static constexpr int64_t GB = int64_t(1024) * 1024 * 1024;

DEFINE_int32(block_size, 16, "slots per block, value must be multiple of 16");
DEFINE_int64(max_cache_size, 5 * GB, "max cache size in bytes, default 5GB");
DEFINE_double(max_memory_utilization,
              0.9,
              "maximum memory utilization allowed, default 0.9");

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

Engine::Engine(const std::vector<torch::Device>& devices) : devices_(devices) {
  CHECK_GT(devices.size(), 0) << "At least one device is required";

  for (const auto device : devices) {
    // all devices should be the same type
    if (device.is_cuda()) {
      // check cuda compute capability
      const auto* properties = at::cuda::getDeviceProperties(device.index());
      const bool is_sm8x = properties->major == 8 && properties->minor >= 0;
      const bool is_sm90 = properties->major == 9 && properties->minor == 0;
      CHECK(is_sm90 || is_sm8x) << "Engine only supports Ampere GPUs or newer.";
      // TODO: add Turing(sm75) support in the near future.
    }
  }

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
    LOG(WARNING) << "Custom kernels are disabled, using generic kernels.";
  }
}

bool Engine::init(const std::string& model_weights_path) {
  if (!init_model(model_weights_path)) {
    LOG(ERROR) << "Failed to initialize model from: " << model_weights_path;
    return false;
  }

  if (!init_kv_cache()) {
    LOG(ERROR) << "Failed to initialize kv cache";
    return false;
  }
  return true;
}

bool Engine::init_model(const std::string& model_weights_path) {
  auto model_loader = ModelLoader::create(model_weights_path);
  LOG(INFO) << "Initializing model from: " << model_weights_path;

  tokenizer_ = model_loader->tokenizer();
  CHECK(tokenizer_ != nullptr);

  args_ = model_loader->model_args();
  quant_args_ = model_loader->quant_args();
  tokenizer_args_ = model_loader->tokenizer_args();
  dtype_ = parse_dtype(args_.dtype(), devices_[0]);
  LOG(INFO) << "Initializing model with dtype: " << dtype_;

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

bool Engine::init_kv_cache() {
  LOG(INFO) << "Initializing kv cache with block size: " << FLAGS_block_size
            << ", max cache size: " << readable_size(FLAGS_max_cache_size)
            << ", max memory utilization: " << FLAGS_max_memory_utilization;

  const int64_t block_size = FLAGS_block_size;

  // init kv cache
  const int world_size = static_cast<int>(workers_.size());
  const int64_t n_heads = args_.n_heads();
  const int64_t n_kv_heads = args_.n_kv_heads().value_or(n_heads);
  const int64_t n_local_kv_heads = n_kv_heads / world_size;
  const int64_t head_dim = args_.hidden_size() / n_heads;
  const auto dtype_size = torch::scalarTypeToTypeMeta(dtype_).itemsize();
  // key + value for all layers
  const int64_t block_size_in_bytes = 2 * block_size * n_local_kv_heads *
                                      head_dim * args_.n_layers() * dtype_size;
  LOG(INFO) << "Block size in bytes: " << readable_size(block_size_in_bytes)
            << ", block_size: " << block_size << ", head_dim: " << head_dim
            << ", n_local_kv_heads: " << n_local_kv_heads
            << ", n_layers: " << args_.n_layers()
            << ", dtype_size: " << dtype_size;

  int64_t num_blocks = 0;
  // use first device to profile memory usage
  const auto& device = workers_[0]->device();
  if (device.is_cpu()) {
    // use max memory cache size for CPU
    LOG(INFO) << "Initializing CPU cache with max cache size: "
              << readable_size(FLAGS_max_cache_size);
    num_blocks = FLAGS_max_cache_size / block_size_in_bytes;
    CHECK_GT(num_blocks, 0) << "Not enough memory for the cache";
  } else if (device.is_cuda()) {
    torch::cuda::synchronize();
    const auto allocated_bytes = memory::max_memory_allocated(device);
    const auto total_memory = memory::total_memory(device);
    LOG(INFO) << device
              << ": allocated GPU memory: " << readable_size(allocated_bytes)
              << ", total GPU memory: " << readable_size(total_memory);

    int64_t max_cache_size =
        static_cast<int64_t>(static_cast<double>(total_memory) *
                             FLAGS_max_memory_utilization) -
        allocated_bytes;
    // apply memory cap from config if it is set
    if (FLAGS_max_cache_size > 0) {
      max_cache_size = std::min(max_cache_size, FLAGS_max_cache_size);
    }
    CHECK_GT(max_cache_size, 0) << "Not enough memory for the cache";
    LOG(INFO) << "Initializing CUDA cache with max cache size: "
              << readable_size(max_cache_size);
    num_blocks = max_cache_size / block_size_in_bytes;
    CHECK_GT(num_blocks, 0) << "Not enough memory for the cache";
  } else {
    CHECK(false) << "Only support CPU and CUDA device for now.";
  }

  // init kv cache for each worker
  const std::vector<int64_t> kv_cache_shape = {
      num_blocks, block_size, n_local_kv_heads, head_dim};
  LOG(INFO) << "Initializing kv cache with shape: [" << kv_cache_shape << "]";

  block_manager_ = std::make_unique<BlockManager>(num_blocks, block_size);

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

OutputParameters Engine::execute_model(const std::vector<Sequence*>& batch) {
  // prepare inputs for workers
  torch::Tensor flatten_token_ids;
  torch::Tensor flatten_positions;
  InputParameters input_params;
  SamplingParameters sampling_params;
  Utils::prepare_inputs(batch,
                        FLAGS_block_size,
                        &flatten_token_ids,
                        &flatten_positions,
                        &input_params,
                        &sampling_params);
  if (workers_.size() == 1) {
    // only one worker, call blocking forward
    auto output = workers_[0]->execute_model(
        flatten_token_ids, flatten_positions, input_params, sampling_params);
    return output;
  }

  // multiple workers, call async forward
  std::vector<folly::SemiFuture<OutputParameters>> futures;
  futures.reserve(workers_.size());
  for (auto& worker : workers_) {
    futures.push_back(worker->execute_model_async(
        flatten_token_ids, flatten_positions, input_params, sampling_params));
  }
  // wait for the all future to complete
  auto results = folly::collectAll(futures).get();
  // return the result from the first worker
  auto first_output = results.front().value();
  return first_output;
}

// TODO
OutputParameters Engine::validate(const std::vector<Sequence*>& batch) {
  torch::Tensor flatten_token_ids;
  torch::Tensor flatten_positions;
  torch::Tensor seq_idxes;
  InputParameters input_params;
  SamplingParameters sampling_params;

  Utils::prepare_validate_inputs(batch,
                                 FLAGS_block_size,
                                 &flatten_token_ids,
                                 &flatten_positions,
                                 &seq_idxes,
                                 &input_params,
                                 &sampling_params);
  if (workers_.size() == 1) {
    auto output = workers_[0]->validate(
        flatten_token_ids, flatten_positions, input_params, sampling_params);
    output.index_select(seq_idxes);
    return output;
  }

  std::vector<folly::SemiFuture<OutputParameters>> futures;
  futures.reserve(workers_.size());
  for (auto& worker : workers_) {
    futures.emplace_back(worker->validate_async(
        flatten_token_ids, flatten_positions, input_params, sampling_params));
  }
  auto results = folly::collectAll(futures).get();
  auto first_output = results.front().value();

  first_output.index_select(seq_idxes);
  return first_output;
}

}  // namespace llm
