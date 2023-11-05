#include <absl/strings/str_split.h>
#include <c10/core/Device.h>
#include <folly/init/Init.h>
#include <gflags/gflags.h>
#include <torch/torch.h>

#include <csignal>
#include <filesystem>
#include <memory>
#include <nlohmann/json.hpp>
#include <thread>

#include "common/logging.h"
#include "common/metrics.h"
#include "engine/engine.h"
#include "grpc_server.h"
#include "handlers/chat_handler.h"
#include "handlers/completion_handler.h"
#include "handlers/models_handler.h"
#include "http_server.h"
#include "scheduler/continuous_batching_scheduler.h"

using namespace llm;

DEFINE_string(model_id, "", "hf model name.");

DEFINE_string(model_path, "", "hf model path to the model file.");

DEFINE_string(device,
              "auto",
              "Device to run the model on, e.g. cpu, cuda:0, cuda:0,cuda:1, or "
              "auto to use all available gpus.");

DEFINE_int32(http_port, 9999, "Port for http server.");
DEFINE_int32(grpc_port, 8888, "Port for grpc server.");

// NOLINTNEXTLINE
static std::atomic<bool> running{true};
void shutdown_handler(int signal) {
  GLOG(WARNING) << "Received signal " << signal << ", stopping server...";
  running.store(false, std::memory_order_relaxed);
}

std::vector<torch::Device> parse_devices(const std::string& device_str) {
  std::vector<torch::Device> devices;
  if (device_str == "auto") {
    // use all available gpus if any
    const auto num_gpus = torch::cuda::device_count();
    if (num_gpus == 0) {
      GLOG(INFO) << "no gpus found, using cpu.";
      return {torch::kCPU};
    }
    devices.reserve(num_gpus);
    for (int i = 0; i < num_gpus; ++i) {
      devices.emplace_back(torch::kCUDA, i);
    }
    GLOG(INFO) << "using " << num_gpus << " gpus.";
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
  GCHECK(!devices.empty()) << "No devices specified.";
  GCHECK(device_types.size() == 1)
      << "All devices must be of the same type. Got: " << FLAGS_device;
  return devices;
}

int main(int argc, char** argv) {
  // glog and glfag will be initialized in folly::init
  folly::Init init(&argc, &argv);
  google::InstallFailureSignalHandler();

  // check if model path exists
  if (!std::filesystem::exists(FLAGS_model_path)) {
    LOG(FATAL) << "Model path " << FLAGS_model_path << " does not exist.";
  }

  if (FLAGS_model_id.empty()) {
    // use last part of the path as model id
    FLAGS_model_id = std::filesystem::path(FLAGS_model_path).filename();
  }

  HttpServer http_server;
  http_server.RegisterURI("/gflags",
                          [](HttpServer::Transport& transport) -> bool {
                            auto gflags = nlohmann::json::array();
                            std::vector<google::CommandLineFlagInfo> flags;
                            google::GetAllFlags(&flags);
                            for (const auto& flag : flags) {
                              nlohmann::json gflag;
                              gflag["name"] = flag.name;
                              gflag["type"] = flag.type;
                              gflag["description"] = flag.description;
                              gflag["value"] = flag.current_value;
                              gflag["default"] = flag.default_value;
                              gflags.push_back(gflag);
                            }
                            return transport.SendString(
                                gflags.dump(/*indent=*/2), "application/json");
                          });
  http_server.RegisterURI(
      "/metrics", [](HttpServer::Transport& transport) -> bool {
        return transport.SendString(Metrics::Instance().GetString());
      });
  http_server.RegisterURI("/health",
                          [](HttpServer::Transport& transport) -> bool {
                            return transport.SendString("Ok\n");
                          });

  if (!http_server.Start(FLAGS_http_port, /*num_threads=*/2)) {
    GLOG(ERROR) << "Failed to start http server on port " << FLAGS_http_port;
    return -1;
  }
  GLOG(INFO) << "Started http server on localhost:" << FLAGS_http_port;

  // parse devices
  const auto devices = parse_devices(FLAGS_device);

  // set the default dtype
  torch::ScalarType dtype{};
  if (devices[0].is_cpu()) {
    // always use float32 on CPU since float16 is not supported
    dtype = torch::kFloat;
    GLOG(INFO) << "Using float32 on CPU.";
  } else {
    dtype = torch::kHalf;
  }

  // create engine
  auto engine = std::make_unique<Engine>(dtype, devices);
  GCHECK(engine->init(FLAGS_model_path));

  // create scheduler and grpc handlers
  auto scheduler = std::make_unique<ContinuousBatchingScheduler>(engine.get());
  auto completion_handler =
      std::make_unique<CompletionHandler>(scheduler.get(), engine.get());
  auto chat_handler =
      std::make_unique<ChatHandler>(scheduler.get(), engine.get());
  auto models_handler = std::make_unique<ModelsHandler>(FLAGS_model_id);

  // start grpc server
  GrpcServer grpc_server(std::move(completion_handler),
                         std::move(chat_handler),
                         std::move(models_handler));
  GrpcServer::Options options;
  options.address = "localhost";
  options.port = FLAGS_grpc_port;

  if (!grpc_server.start(options)) {
    GLOG(ERROR) << "failed to start grpc server";
    return -1;
  }
  // install graceful shutdown handler
  (void)signal(SIGINT, shutdown_handler);
  (void)signal(SIGTERM, shutdown_handler);

  const auto timeout = absl::Milliseconds(500);
  while (running.load(std::memory_order_relaxed)) {
    // move scheduler forward
    scheduler->step(timeout);
  }

  // stop grpc server and http server
  grpc_server.stop();
  http_server.Stop();

  return 0;
}
