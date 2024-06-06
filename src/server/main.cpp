#include <absl/strings/numbers.h>
#include <absl/strings/str_split.h>
#include <folly/init/Init.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <csignal>
#include <filesystem>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>

#include "common/metrics.h"
#include "grpc_server.h"
#include "handlers/chat_handler.h"
#include "handlers/completion_handler.h"
#include "handlers/llm_handler.h"
#include "handlers/models_handler.h"
#include "http_server.h"
using namespace llm;

DEFINE_int32(http_port, 9999, "Port for http server.");
DEFINE_int32(grpc_port, 8888, "Port for grpc server.");

DEFINE_string(model_id, "", "hf model name.");

DEFINE_string(model_path, "", "hf model path to the model file.");

DEFINE_string(device,
              "auto",
              "Device to run the model on, e.g. cpu, cuda:0, cuda:0,cuda:1, or "
              "auto to use all available gpus.");

DEFINE_string(draft_model_path, "", "draft hf model path to the model file.");

DEFINE_string(
    draft_device,
    "cuda:0",
    "Device to run the draft model on, e.g. cpu, cuda:0, cuda:0,cuda:1, or "
    "auto to use all available gpus.");

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
             2048,
             "max sequence length used to capture cuda graphs");

DEFINE_string(cuda_graph_batch_sizes,
              "",
              "batch sizes to capture cuda graphs, comma separated list");

DEFINE_string(
    draft_cuda_graph_batch_sizes,
    "",
    "batch sizes to capture cuda graphs for draft model, comma separated list");

DEFINE_int32(max_tokens_per_batch, 512, "max number of tokens per batch");

DEFINE_int32(max_seqs_per_batch, 128, "max number of sequences per batch");

DEFINE_int32(num_speculative_tokens, 0, "number of speculative tokens");

// NOLINTNEXTLINE
static std::atomic<uint32_t> signal_received{0};
void shutdown_handler(int signal) {
  // TODO: gracefully shutdown the server
  LOG(WARNING) << "Received signal " << signal << ", stopping server...";
  exit(1);
}

std::optional<std::vector<uint32_t>> parse_batch_sizes(
    const std::string& batch_sizes_str) {
  if (batch_sizes_str.empty() || batch_sizes_str == "auto") {
    return std::nullopt;
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
  if (sizes_set.empty()) {
    return std::nullopt;
  }
  return std::vector<uint32_t>{sizes_set.begin(), sizes_set.end()};
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
  http_server.register_uri("/gflags",
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
                             return transport.send_string(
                                 gflags.dump(/*indent=*/2), "application/json");
                           });
  http_server.register_uri(
      "/metrics", [](HttpServer::Transport& transport) -> bool {
        return transport.send_string(Metrics::Instance().GetString());
      });
  http_server.register_uri("/health",
                           [](HttpServer::Transport& transport) -> bool {
                             return transport.send_string("OK\n");
                           });

  // Create LLMHandler
  LLMHandler::Options options;
  options.model_path(FLAGS_model_path)
      .devices(FLAGS_device)
      .draft_model_path(FLAGS_draft_model_path)
      .draft_devices(FLAGS_draft_device)
      .block_size(FLAGS_block_size)
      .max_cache_size(FLAGS_max_cache_size)
      .max_memory_utilization(FLAGS_max_memory_utilization)
      .enable_prefix_cache(FLAGS_enable_prefix_cache)
      .enable_cuda_graph(FLAGS_enable_cuda_graph)
      .cuda_graph_max_seq_len(FLAGS_cuda_graph_max_seq_len)
      .cuda_graph_batch_sizes(parse_batch_sizes(FLAGS_cuda_graph_batch_sizes))
      .draft_cuda_graph_batch_sizes(
          parse_batch_sizes(FLAGS_draft_cuda_graph_batch_sizes))
      .max_tokens_per_batch(FLAGS_max_tokens_per_batch)
      .max_seqs_per_batch(FLAGS_max_seqs_per_batch)
      .num_speculative_tokens(FLAGS_num_speculative_tokens);

  auto llm_handler = std::make_unique<LLMHandler>(options);
  llm_handler->start();

  // supported models
  std::vector<std::string> models = {FLAGS_model_id};
  auto completion_handler =
      std::make_unique<CompletionHandler>(llm_handler.get(), models);
  auto chat_handler = std::make_unique<ChatHandler>(llm_handler.get(), models);
  auto models_handler = std::make_unique<ModelsHandler>(models);

  // start grpc server
  GrpcServer grpc_server(std::move(completion_handler),
                         std::move(chat_handler),
                         std::move(models_handler));
  GrpcServer::Options grpc_options;
  grpc_options.address = "0.0.0.0";
  grpc_options.port = FLAGS_grpc_port;

  if (!grpc_server.start(grpc_options)) {
    LOG(ERROR) << "failed to start grpc server on port " << FLAGS_grpc_port;
    return -1;
  }

  if (!http_server.start(FLAGS_http_port, /*num_threads=*/2)) {
    LOG(ERROR) << "Failed to start http server on port " << FLAGS_http_port;
    return -1;
  }

  // install graceful shutdown handler
  (void)signal(SIGINT, shutdown_handler);
  (void)signal(SIGTERM, shutdown_handler);

  while (signal_received.load(std::memory_order_relaxed) == 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

  // stop grpc server and http server
  grpc_server.stop();
  http_server.stop();
  llm_handler->stop();
  return 0;
}
