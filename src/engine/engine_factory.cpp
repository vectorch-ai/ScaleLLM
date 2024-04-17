#include "engine_factory.h"

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
            "Enable CUDAGraph to optimize model execution.");

DECLARE_int32(num_speculative_tokens);

namespace llm {
namespace {

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
        .enable_cuda_graph(FLAGS_enable_cuda_graph)
        .num_speculative_tokens(FLAGS_num_speculative_tokens);
    auto engine = std::make_unique<SpeculativeEngine>(options);
    CHECK(engine->init(model_path, draft_model_path));
    return engine;
  }

  LLMEngine::Options options;
  options.devices(devices)
      .block_size(FLAGS_block_size)
      .max_cache_size(FLAGS_max_cache_size)
      .max_memory_utilization(FLAGS_max_memory_utilization)
      .enable_prefix_cache(FLAGS_enable_prefix_cache)
      .enable_cuda_graph(FLAGS_enable_cuda_graph);

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
      .enable_prefix_cache(FLAGS_enable_prefix_cache)
      .enable_cuda_graph(FLAGS_enable_cuda_graph);
  auto engine = std::make_unique<LLMEngine>(options);
  CHECK(engine->init(model_path));
  return engine;
}

}  // namespace llm
