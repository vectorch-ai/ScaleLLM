#pragma once

#include <folly/Function.h>

#include <functional>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "chat_template/chat_template.h"
#include "common/concurrent_queue.h"
#include "engine/engine.h"
#include "request/output.h"
#include "sampling_params.h"
#include "scheduler/continuous_scheduler.h"

namespace llm {

// callback function for output, return true to continue, false to stop/cancel
using OutputCallback = std::function<bool(RequestOutput output)>;

// NOLINTNEXTLINE
class VLMHandler {
 public:
  struct Options {
    DEFINE_ARG(std::string, model_path);

    DEFINE_ARG(std::optional<std::string>, devices);

    // the number of slots per block, default 16, value must be multiple of 16
    DEFINE_ARG(int32_t, block_size) = 16;

    // the maximum cache size in bytes, default 10GB
    DEFINE_ARG(int64_t, max_cache_size) = static_cast<int64_t>(10) * 1024 *
                                          1024 * 1024;

    // maximum memory utilization allowed, default 0.9
    DEFINE_ARG(double, max_memory_utilization) = 0.9;

    // enable prefix cache
    DEFINE_ARG(bool, enable_prefix_cache) = true;

    // enable cuda graph
    DEFINE_ARG(bool, enable_cuda_graph) = true;

    // max sequence length used to capture cuda graphs
    DEFINE_ARG(int64_t, cuda_graph_max_seq_len) = 2048;

    // batch sizes to capture cuda graphs
    DEFINE_ARG(std::optional<std::vector<uint32_t>>, cuda_graph_batch_sizes);

    // the maximum number of tokens per batch
    DEFINE_ARG(int32_t, max_tokens_per_batch) = 256;

    // the maximum number of sequences per batch
    DEFINE_ARG(int32_t, max_seqs_per_batch) = 64;

    // the number of threads to use for handling requests
    DEFINE_ARG(size_t, num_handling_threads) = 4;

    DEFINE_ARG(std::string, image_input_type) = "pixel_values";

    DEFINE_ARG(int64_t, image_token_id) = 32000;

    DEFINE_ARG(std::string, image_input_shape) = "1,3,336,336";

    DEFINE_ARG(int32_t, image_feature_size) = 576;
  };

  VLMHandler(const Options& options);

  ~VLMHandler();

  // schedule a request, the engine will execute the request asynchronously
  // and call the callback with output when the request is done
  // the callback will be called multiple times if the request is a streaming
  // request
  std::future<bool> schedule_async(torch::Tensor image,
                                   std::string prompt,
                                   SamplingParams sp,
                                   Priority priority,
                                   bool stream,
                                   OutputCallback callback);

  // start the handling loop
  void start();

  // stop the engine
  void stop();

  // run until complete, blocking call
  void run_until_complete();

  std::vector<int32_t> encode(const std::string& text);

  std::string decode(const std::vector<int32_t>& tokens,
                     bool skip_special_tokens);

  // release underlying resources
  void reset();

  const Options& options() const { return options_; }

 private:
  using Task = folly::Function<void(size_t tid)>;
  std::unique_ptr<Request> create_request(size_t tid,
                                          torch::Tensor image,
                                          std::string prompt,
                                          const SamplingParams& sp,
                                          Priority priority,
                                          bool stream,
                                          OutputCallback callback);

  std::future<bool> schedule(torch::Tensor image,
                             std::string prompt,
                             SamplingParams sp,
                             Priority priority,
                             bool stream,
                             OutputCallback callback);

  void handling_loop(size_t tid);

  const Options options_;

  std::unique_ptr<Engine> engine_;

  std::unique_ptr<Scheduler> scheduler_;

  // model args
  ModelArgs model_args_;

  // thread pool for handling requests
  std::vector<std::thread> handling_threads_;

  // queue for tasks
  ConcurrentQueue<Task> queue_;

  // we don't know if tokenizer is thread safe, so we create one for each thread
  // for now
  std::vector<std::unique_ptr<Tokenizer>> tokenizers_;

  // thread for moving forward the scheduler
  std::thread loop_thread_;

  // flag to stop the loop
  std::atomic_bool stoped_{false};

  // flag to indicate if the handler is running
  std::atomic_bool running_{false};
};

}  // namespace llm
