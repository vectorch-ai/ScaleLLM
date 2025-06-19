#include "llm_engine.h"

#include <ATen/cuda/CUDAContext.h>
#include <glog/logging.h>
#include <sys/sysinfo.h>

#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <memory>

#include "common/metrics.h"
#include "common/pretty_print.h"
#include "model_loader/model_loader.h"
#include "model_parallel/parallel_args.h"
#include "models/model_args.h"
#include "worker.h"

DEFINE_COUNTER(prepare_input_latency_seconds,
               "Latency of preparing input in seconds");

namespace llm {
namespace {
const std::vector<uint32_t> kDefaultBatchSizesForCudaGraph =
    {1, 2, 4, 8, 16, 24, 32, 48, 64};

torch::ScalarType parse_dtype(const std::string& dtype_str,
                              const torch::Device& device) {
  if (device.is_cpu()) {
    // cpu only supports float32 for now
    return torch::kFloat32;
  }

  if (boost::iequals(dtype_str, "half") ||
      boost::iequals(dtype_str, "float16")) {
    return torch::kFloat16;
  }
  if (boost::iequals(dtype_str, "bfloat16")) {
    return torch::kBFloat16;
  }
  if ((boost::iequals(dtype_str, "float") ||
       boost::iequals(dtype_str, "float32"))) {
    // cuda only supports float16 and bfloat16 for now
    return torch::kFloat16;
  }

  if (dtype_str.empty() || boost::iequals(dtype_str, "auto")) {
    return torch::kFloat16;
  }
  CHECK(false) << "Unsupported dtype: " << dtype_str << " on device " << device;
}
}  // namespace

LLMEngine::LLMEngine(const Options& options) : options_(options) {
  const auto& devices = options_.devices();
  CHECK_GT(devices.size(), 0) << "At least one device is required";

  const auto device_type = devices[0].type();
  for (const auto device : devices) {
    CHECK_EQ(device.type(), device_type)
        << "All devices should be the same type";

    if (device.is_cuda()) {
      // check cuda compute capability
      const auto* properties = at::cuda::getDeviceProperties(device.index());
      CHECK(properties->major >= 8) << "Only supports Ampere GPUs or newer.";
    }
  }

  // initialize process groups if there are multiple devices
  if (devices.size() > 1) {
    // create a process group for each device if there are multiple gpus
    process_groups_ = ProcessGroup::create_process_groups(devices);
  }

  // sort cuda graph batch sizes
  if (options_.enable_cuda_graph()) {
    batch_sizes_ = options_.cuda_graph_batch_sizes().value_or(
        kDefaultBatchSizesForCudaGraph);
    std::sort(batch_sizes_.begin(), batch_sizes_.end());
  }

  // create a worker for each device
  ModelRunner::Options runner_options;
  runner_options.block_size(options_.block_size())
      .num_decoding_tokens(options_.num_decoding_tokens())
      .cuda_graph_max_seq_len(options_.cuda_graph_max_seq_len())
      .cuda_graph_batch_sizes(batch_sizes_);

  const int32_t world_size = static_cast<int32_t>(devices.size());
  for (size_t i = 0; i < devices.size(); ++i) {
    const int32_t rank = static_cast<int32_t>(i);
    ProcessGroup* pg = world_size > 1 ? process_groups_[i].get() : nullptr;
    ParallelArgs parallel_args(rank, world_size, pg);
    workers_.emplace_back(
        std::make_unique<Worker>(parallel_args, devices[i], runner_options));
  }

  if (workers_.size() > 1) {
    // test process group
    std::vector<folly::SemiFuture<folly::Unit>> futures;
    futures.reserve(workers_.size());
    for (auto& worker : workers_) {
      futures.emplace_back(worker->process_group_test_async());
    }
    // wait up to 4 seconds for all futures to complete
    folly::collectAll(futures).within(std::chrono::seconds(4)).get();
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
  if (!capture_cuda_graphs()) {
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
  n_local_kv_heads_ = std::max<int64_t>(1, n_kv_heads / world_size);
  head_dim_ = args_.head_dim();
  dtype_ = parse_dtype(args_.dtype(), options_.devices()[0]);

  // key + value for all layers
  LOG(INFO) << "Block info, block_size: " << options_.block_size()
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

bool LLMEngine::capture_cuda_graphs() {
  if (!options_.enable_cuda_graph()) {
    return true;
  }

  LOG(INFO) << "Capturing CUDA graphs: num_decoding_tokens: "
            << options_.num_decoding_tokens()
            << ", batch sizes: " << batch_sizes_;

  for (const auto batch_size : batch_sizes_) {
    std::vector<folly::SemiFuture<folly::Unit>> futures;
    futures.reserve(workers_.size());
    for (auto& worker : workers_) {
      futures.emplace_back(worker->capture_cuda_graph_async(batch_size));
    }
    // wait up to 4 seconds for all futures to complete
    folly::collectAll(futures).within(std::chrono::seconds(4)).get();
  }
  return true;
}

int64_t LLMEngine::profile_memory_for_kv_cache() {
  const int64_t max_cache_size = options_.max_cache_size();
  const double max_memory_utilization = options_.max_memory_utilization();

  const auto& device = workers_[0]->device();
  if (device.is_cpu()) {
    // get cpu available memory and total memory
    struct sysinfo info;
    int err = sysinfo(&info);
    if (err != 0) {
      LOG(ERROR) << "Initializing CPU cache failure.";
    }
    int64_t available_memory = info.freeram;
    int64_t total_memory = info.totalram;

    int64_t smallest_available_memory = std::numeric_limits<int64_t>::max();
    // apply memory cap from config if it is set
    if (max_memory_utilization < 1.0) {
      const int64_t buffer_memory =
          total_memory * (1.0 - max_memory_utilization);
      available_memory -= buffer_memory;
    }
    if (max_cache_size > 0) {
      available_memory = std::min(available_memory, max_cache_size);
    }
    smallest_available_memory =
        std::min(smallest_available_memory, available_memory);

    auto cache_size = std::max(smallest_available_memory, int64_t(0));
    LOG(INFO) << "Initializing CPU cache with cache size: "
              << readable_size(cache_size);
    return cache_size;
  }
  CHECK(device.is_cuda()) << "Only support CPU and CUDA device for now.";

  // call worker to profile memory usage
  std::vector<folly::SemiFuture<std::tuple<int64_t, int64_t>>> futures;
  futures.reserve(workers_.size());
  for (auto& worker : workers_) {
    futures.push_back(worker->profile_device_memory_async());
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

    LOG(INFO) << "Using max_memory_utilization: " << max_memory_utilization
              << ", max_cache_size: " << readable_size(max_cache_size);
    // apply memory cap from config if it is set
    if (max_memory_utilization < 1.0) {
      const int64_t buffer_memory =
          total_memory * (1.0 - max_memory_utilization);
      available_memory -= buffer_memory;
    }
    if (max_cache_size > 0) {
      available_memory = std::min(available_memory, max_cache_size);
    }
    smallest_available_memory =
        std::min(smallest_available_memory, available_memory);
  }
  return std::max(smallest_available_memory, int64_t(0));
}

bool LLMEngine::init_kv_cache(int64_t n_blocks) {
  CHECK_GT(n_blocks, 0) << "no memory for kv cache";
  const int32_t block_size = options_.block_size();

  // init kv cache for each worker
  LOG(INFO) << "Initializing kv cache with shape: [" << n_blocks << ", "
            << block_size << ", " << n_local_kv_heads_ << ", " << head_dim_
            << "]";

  // initialize block manager
  BlockManager::Options options;
  options.num_blocks(n_blocks)
      .block_size(block_size)
      .enable_prefix_cache(options_.enable_prefix_cache());
  block_manager_ = std::make_unique<BlockManager>(options);

  // init kv cache for each worker in parallel
  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(workers_.size());
  for (auto& worker : workers_) {
    futures.push_back(worker->init_kv_cache_async(
        n_blocks, block_size, n_local_kv_heads_, head_dim_));
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
  uint32_t adjusted_batch_size = 0;
  if (options_.enable_cuda_graph()) {
    // find the closest batch size in the captured graph
    const auto it = std::lower_bound(
        batch_sizes_.begin(), batch_sizes_.end(), batch.size());
    if (it != batch_sizes_.end()) {
      adjusted_batch_size = *it;
    }
  }

  Timer timer;
  auto model_inputs = batch.prepare_model_input(options_.num_decoding_tokens(),
                                                adjusted_batch_size);
  COUNTER_ADD(prepare_input_latency_seconds, timer.elapsed_seconds());

  if (!model_inputs.token_ids.defined()) {
    // empty input, just return
    return {};
  }

  std::vector<folly::SemiFuture<std::optional<ModelOutput>>> futures;
  futures.reserve(workers_.size());
  for (auto& worker : workers_) {
    futures.emplace_back(worker->execute_model_async(model_inputs));
  }
  // wait for the all future to complete
  auto results = folly::collectAll(futures).get();
  // return the result from the driver
  auto model_output = results.front().value();
  DCHECK(model_output.has_value()) << "Failed to execute model";
  batch.process_sample_output(model_output.value().sample_output);
  return model_output.value();
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
  const int32_t block_size = options_.block_size();
  const int64_t block_size_in_bytes =
      block_size * kv_cache_slot_size_in_bytes();
  return cache_size_in_bytes / block_size_in_bytes;
}

}  // namespace llm
