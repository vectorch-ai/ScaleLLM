#include "engine.h"

#include <memory>

#include "common/pretty_print.h"
#include "memory/memory.h"
#include "model_loader/model_loader.h"
#include "models/args.h"
#include "request/request.h"
#include "tokenizer/sentencepiece_tokenizer.h"
#include "utils.h"
#include "worker.h"

static constexpr int64_t GB = int64_t(1024) * 1024 * 1024;

DEFINE_int32(max_seq_len, 256, "Maximum sequence length.");

DEFINE_int32(block_size, 16, "slots per block, value must be [8, 16, 32]");
DEFINE_int64(max_cache_size, 5 * GB, "max cache size in bytes, default 5GB");
DEFINE_double(max_memory_utilization,
              0.9,
              "maximum memory utilization allowed, default 0.9");

namespace llm {

Engine::Engine(torch::ScalarType dtype,
               const std::vector<torch::Device>& devices)
    : dtype_(dtype), devices_(devices) {
  CHECK_GT(devices.size(), 0) << "At least one device is required";

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
    workers_.emplace_back(
        std::make_unique<Worker>(parallel_args, dtype, devices[i]));
  }
}

bool Engine::init(const std::string& model_weights_path,
                  const std::string& tokenizer_path) {
  // load tokenizer
  // TODO: support other tokenizers
  tokenizer_ = std::make_unique<SentencePieceTokenizer>(tokenizer_path);

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

  args_ = model_loader->model_args();
  if (args_.vocab_size() == -1) {
    args_.vocab_size(static_cast<int64_t>(tokenizer_->vocab_size()));
  }
  // TODO: remove this two from model args
  args_.max_seq_len(FLAGS_max_seq_len);

  const auto& quant_args = model_loader->quant_args();

  // init model for each worker in parallel
  // multiple workers, call async init
  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(workers_.size());
  for (auto& worker : workers_) {
    futures.push_back(worker->init_model_async(args_, quant_args));
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
  const int64_t block_size_in_bytes = int64_t(2) * block_size *
                                      n_local_kv_heads * head_dim *
                                      args_.n_layers() * dtype_size;
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
    LOG(INFO) << "Initializing CUDA cache with max cache size: "
              << readable_size(max_cache_size);
    num_blocks = max_cache_size / block_size_in_bytes;
    CHECK_GT(num_blocks, 0) << "Not enough memory for the cache";
  } else {
    CHECK(false) << "Only support CPU and CUDA device for now.";
  }

  LOG(INFO) << "Initializing kv cache with num blocks: " << num_blocks
            << ", block size: " << block_size;

  // init kv cache for each worker
  const int64_t x = 16 / dtype_size;
  const std::vector<int64_t> key_cache_shape = {
      num_blocks, n_local_kv_heads, head_dim / x, block_size, x};
  const std::vector<int64_t> value_cache_shape = {
      num_blocks, n_local_kv_heads, head_dim, block_size};
  LOG(INFO) << "Initializing kv cache with key shape: [" << key_cache_shape
            << "], value shape: [" << value_cache_shape << "]";

  // init kv cache for each worker in parallel
  std::vector<folly::SemiFuture<bool>> futures;
  futures.reserve(workers_.size());
  for (auto& worker : workers_) {
    futures.push_back(
        worker->init_kv_cache_async(key_cache_shape, value_cache_shape));
  }
  // wait for all futures to complete
  auto results = folly::collectAll(futures).get();
  for (const auto& result : results) {
    if (!result.value()) {
      return false;
    }
  }

  block_manager_ = std::make_unique<BlockManager>(num_blocks, block_size);
  return true;
}

OutputParameters Engine::execute_model(const std::vector<Sequence*>& batch) {
  // prepare inputs for workers
  torch::Tensor input_token_ids;
  torch::Tensor input_positions;
  // track the sequence indices in the batch
  torch::Tensor seq_indices;
  InputParameters input_params;
  SamplingParameters sampling_params;
  Utils::prepare_inputs(batch,
                        FLAGS_block_size,
                        &input_token_ids,
                        &input_positions,
                        &seq_indices,
                        &input_params,
                        &sampling_params);
  if (workers_.size() == 1) {
    // only one worker, call blocking forward
    auto output = workers_[0]->execute_model(
        input_token_ids, input_positions, input_params, sampling_params);
    // mapping outout back to the original request order in the batch
    output.index_select(seq_indices);
    return output;
  }

  // multiple workers, call async forward
  std::vector<folly::SemiFuture<OutputParameters>> futures;
  futures.reserve(workers_.size());
  for (auto& worker : workers_) {
    futures.push_back(worker->execute_model_async(
        input_token_ids, input_positions, input_params, sampling_params));
  }
  // wait for the all future to complete
  auto results = folly::collectAll(futures).get();
  // return the result from the first worker
  auto first_output = results.front().value();
  // mapping output back to the original request order in the batch
  first_output.index_select(seq_indices);
  return first_output;
}

}  // namespace llm
