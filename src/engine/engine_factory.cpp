#include "engine_factory.h"

#include <absl/strings/str_split.h>

#include "engine/llm_engine.h"
#include "speculative/speculative_engine.h"

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
    auto engine = std::make_unique<SpeculativeEngine>(devices, draft_devices);
    CHECK(engine->init(model_path, draft_model_path));
    return engine;
  }

  auto engine = std::make_unique<LLMEngine>(devices);
  CHECK(engine->init(model_path));
  return engine;
}

std::unique_ptr<Engine> EngineFactory::create(const std::string& model_path,
                                              const std::string& devices_str) {
  const auto devices = parse_devices(devices_str);
  LOG(INFO) << "Using devices: " << to_string(devices);
  auto engine = std::make_unique<LLMEngine>(devices);
  CHECK(engine->init(model_path));
  return engine;
}

}  // namespace llm
