#include "engine_factory.h"

#include <absl/strings/numbers.h>
#include <absl/strings/str_split.h>

#include "engine/llm_engine.h"
#include "speculative/speculative_engine.h"

static constexpr int64_t GB = int64_t(1024) * 1024 * 1024;

DEFINE_int32(block_size, 16, "slots per block, value must be multiple of 16");

DEFINE_int64(max_cache_size, 10 * GB, "max cache size in bytes, default 10GB");
DEFINE_double(max_memory_utilization,
              0.9,
              "maximum memory utilization allowed, default 0.9");

DEFINE_bool(enable_prefix_cache,
            true,
            "enable the prefix cache for the block manager");

DEFINE_bool(enable_cuda_graph,
            true,
            "Enable CUDA Graph to optimize model execution.");

DEFINE_int64(cuda_graph_max_seq_len,
             4096,
             "max sequence length used to capture cuda graphs");
DEFINE_string(cuda_graph_batch_sizes,
              "auto",
              "batch sizes to capture cuda graphs, comma separated list");
DEFINE_string(
    draft_cuda_graph_batch_sizes,
    "auto",
    "batch sizes to capture cuda graphs for draft model, comma separated list");

DECLARE_int32(num_speculative_tokens);

namespace llm {
namespace {

const std::vector<uint32_t> kDefaultBatchSizesForCudaGraph = {
    1,  2,  3,  4,  5,  6,  7,  8,   16,  24,  32, 40,
    48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128};

std::vector<uint32_t> parse_cuda_graph_batch_sizes(
    const std::string& batch_sizes_str,
    const std::vector<torch::Device>& devices) {
  if (batch_sizes_str == "auto") {
    if (devices.size() == 1 && devices[0].is_cuda()) {
      // use default batch sizes for cuda graph
      return kDefaultBatchSizesForCudaGraph;
    }

    // It is a known issue (https://github.com/vectorch-ai/ScaleLLM/issues/131)
    // that CUDA graph capture may occasionally become stuck with multiple gpus.
    // disable cuda graph for multi-gpus by default
    return {};
  }

  // parse device string
  const std::vector<std::string> size_strs =
      absl::StrSplit(batch_sizes_str, ',');
  // remove duplicates
  std::unordered_set<uint32_t> sizes_set;
  for (const auto& size_str : size_strs) {
    uint32_t batch_size = 0;
    if (!absl::SimpleAtoi(size_str, &batch_size)) {
      LOG(ERROR) << "Failed to parse batch size: " << size_str;
      continue;
    }
    sizes_set.insert(batch_size);
  }
  return {sizes_set.begin(), sizes_set.end()};
}

std::vector<torch::Device> parse_devices(const std::string& device_str) {
  std::vector<torch::Device> devices;
  if (device_str == "auto") {
    // use all available gpus if any
    const auto num_gpus = torch::cuda::device_count();
    if (num_gpus == 0) {
      LOG(INFO) << "no gpus found, using cpu.";
      return {torch::kCPU};
    }
    devices.reserve(num_gpus);
    for (int i = 0; i < num_gpus; ++i) {
      devices.emplace_back(torch::kCUDA, i);
    }
    return devices;
  }

  // parse device string
  const std::vector<std::string> device_strs = absl::StrSplit(device_str, ',');
  std::set<torch::DeviceType> device_types;
  devices.reserve(device_strs.size());
  for (const auto& device_str : device_strs) {
    devices.emplace_back(device_str);
    device_types.insert(devices.back().type());
  }
  CHECK(!devices.empty()) << "No devices specified.";
  CHECK(device_types.size() == 1)
      << "All devices must be of the same type. Got: " << device_str;
  return devices;
}

std::string to_string(const std::vector<torch::Device>& devices) {
  std::stringstream ss;
  for (size_t i = 0; i < devices.size(); ++i) {
    const auto& device = devices[i];
    if (i == 0) {
      ss << device;
    } else {
      ss << "," << device;
    }
  }
  return ss.str();
}
}  // namespace

std::unique_ptr<Engine> EngineFactory::create(
    const std::string& model_path,
    const std::string& devices_str,
    const std::string& draft_model_path,
    const std::string& draft_devices_str) {
  // parse devices
  const auto devices = parse_devices(devices_str);
  LOG(INFO) << "Using devices: " << to_string(devices);

  if (!draft_model_path.empty()) {
    const auto draft_devices = parse_devices(draft_devices_str);
    LOG(INFO) << "Using draft devices: " << to_string(draft_devices);
    SpeculativeEngine::Options options;
    options.devices(devices)
        .draft_devices(draft_devices)
        .block_size(FLAGS_block_size)
        .max_cache_size(FLAGS_max_cache_size)
        .max_memory_utilization(FLAGS_max_memory_utilization)
        .enable_prefix_cache(FLAGS_enable_prefix_cache)
        .num_speculative_tokens(FLAGS_num_speculative_tokens);
    if (FLAGS_enable_cuda_graph) {
      LOG(INFO) << "Using cuda graph optimization, batch sizes: "
                << FLAGS_cuda_graph_batch_sizes;
      const auto batch_sizes =
          parse_cuda_graph_batch_sizes(FLAGS_cuda_graph_batch_sizes, devices);
      const auto draft_batch_sizes = parse_cuda_graph_batch_sizes(
          FLAGS_draft_cuda_graph_batch_sizes, draft_devices);
      options.cuda_graph_max_seq_len(FLAGS_cuda_graph_max_seq_len)
          .cuda_graph_batch_sizes(batch_sizes)
          .draft_cuda_graph_batch_sizes(draft_batch_sizes);
    }
    auto engine = std::make_unique<SpeculativeEngine>(options);
    CHECK(engine->init(model_path, draft_model_path));
    return engine;
  }

  LLMEngine::Options options;
  options.devices(devices)
      .block_size(FLAGS_block_size)
      .max_cache_size(FLAGS_max_cache_size)
      .max_memory_utilization(FLAGS_max_memory_utilization)
      .enable_prefix_cache(FLAGS_enable_prefix_cache);
  if (FLAGS_enable_cuda_graph) {
    LOG(INFO) << "Using cuda graph optimization, batch sizes: "
              << FLAGS_cuda_graph_batch_sizes;
    const auto batch_sizes =
        parse_cuda_graph_batch_sizes(FLAGS_cuda_graph_batch_sizes, devices);
    options.cuda_graph_max_seq_len(FLAGS_cuda_graph_max_seq_len)
        .cuda_graph_batch_sizes(batch_sizes);
  }

  auto engine = std::make_unique<LLMEngine>(options);
  CHECK(engine->init(model_path));
  return engine;
}

std::unique_ptr<Engine> EngineFactory::create(const std::string& model_path,
                                              const std::string& devices_str) {
  const auto devices = parse_devices(devices_str);
  LOG(INFO) << "Using devices: " << to_string(devices);

  LLMEngine::Options options;
  options.devices(devices)
      .block_size(FLAGS_block_size)
      .max_cache_size(FLAGS_max_cache_size)
      .max_memory_utilization(FLAGS_max_memory_utilization)
      .enable_prefix_cache(FLAGS_enable_prefix_cache);
  if (FLAGS_enable_cuda_graph) {
    LOG(INFO) << "Using cuda graph optimization, batch sizes: "
              << FLAGS_cuda_graph_batch_sizes;
    const auto batch_sizes =
        parse_cuda_graph_batch_sizes(FLAGS_cuda_graph_batch_sizes, devices);
    options.cuda_graph_max_seq_len(FLAGS_cuda_graph_max_seq_len)
        .cuda_graph_batch_sizes(batch_sizes);
  }
  auto engine = std::make_unique<LLMEngine>(options);
  CHECK(engine->init(model_path));
  return engine;
}

}  // namespace llm
