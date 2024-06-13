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

using BatchOutputCallback =
    std::function<bool(size_t index, RequestOutput output)>;

class BatchScheduleTask {
 public:
  BatchScheduleTask(std::unique_ptr<std::vector<std::future<bool>>> futures)
      : futures_(std::move(futures)) {}

  void wait() {
    for (const auto& future : *futures_) {
      future.wait();
    }
  }

  std::vector<bool> get() {
    std::vector<bool> results;
    results.reserve(futures_->size());
    for (auto& future : *futures_) {
      results.push_back(future.get());
    }
    return results;
  }

 private:
  // use unique_ptr to workaround 'result type must be constructible from input
  // type' error
  std::unique_ptr<std::vector<std::future<bool>>> futures_;
};

class ScheduleTask {
 public:
  ScheduleTask(std::future<bool> future) : future_(std::move(future)) {}

  void wait() { future_.wait(); }

  bool get() { return future_.get(); }

 private:
  std::future<bool> future_;
};

// NOLINTNEXTLINE
class LLMHandler {
 public:
  struct Options {
    DEFINE_ARG(std::string, model_path);

    DEFINE_ARG(std::optional<std::string>, devices);

    DEFINE_ARG(std::optional<std::string>, draft_model_path);

    DEFINE_ARG(std::optional<std::string>, draft_devices);

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

    // batch sizes to capture cuda graphs for draft model
    DEFINE_ARG(std::optional<std::vector<uint32_t>>,
               draft_cuda_graph_batch_sizes);

    // the maximum number of tokens per batch
    DEFINE_ARG(int32_t, max_tokens_per_batch) = 256;

    // the maximum number of sequences per batch
    DEFINE_ARG(int32_t, max_seqs_per_batch) = 64;

    // the number of speculative tokens per step
    DEFINE_ARG(int32_t, num_speculative_tokens) = 0;

    // the number of threads to use for handling requests
    DEFINE_ARG(size_t, num_handling_threads) = 4;
  };

  LLMHandler(const Options& options);

  ~LLMHandler();

  // schedule a request, the engine will execute the request asynchronously
  // and call the callback with output when the request is done
  // the callback will be called multiple times if the request is a streaming
  // request
  ScheduleTask schedule_async(std::string prompt,
                              SamplingParams sp,
                              Priority priority,
                              bool stream,
                              OutputCallback callback);

  ScheduleTask schedule_chat_async(std::vector<Message> messages,
                                   SamplingParams sp,
                                   Priority priority,
                                   bool stream,
                                   OutputCallback callback);

  // batch version
  BatchScheduleTask schedule_batch_async(std::vector<std::string> prompts,
                                         std::vector<SamplingParams> sp,
                                         Priority priority,
                                         bool stream,
                                         BatchOutputCallback callback);

  BatchScheduleTask schedule_chat_batch_async(
      std::vector<std::vector<Message>> conversations,
      std::vector<SamplingParams> sp,
      Priority priority,
      bool stream,
      BatchOutputCallback callback);

  // start the handling loop
  void start();

  // stop the engine
  void stop();

  // run until complete, blocking call
  void run_until_complete();

  // helper functions exposed for in python
  // apply the chat template to the conversation and return the result
  std::optional<std::string> apply_chat_template(
      const std::vector<Message>& conversation);

  std::vector<int32_t> encode(const std::string& text);

  std::string decode(const std::vector<int32_t>& tokens,
                     bool skip_special_tokens);

  // release underlying resources
  void reset();

 private:
  using Task = folly::Function<void(size_t tid)>;
  std::unique_ptr<Request> create_request(size_t tid,
                                          std::string prompt,
                                          const SamplingParams& sp,
                                          Priority priority,
                                          bool stream,
                                          OutputCallback callback);

  std::unique_ptr<Request> create_chat_request(
      size_t tid,
      const std::vector<Message>& messages,
      const SamplingParams& sp,
      Priority priority,
      bool stream,
      OutputCallback callback);

  std::future<bool> schedule(std::string prompt,
                             SamplingParams sp,
                             Priority priority,
                             bool stream,
                             OutputCallback callback);

  std::future<bool> schedule(std::vector<Message> messages,
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

  // chat template instance
  std::unique_ptr<ChatTemplate> chat_template_;

  // thread for moving forward the scheduler
  std::thread loop_thread_;

  // flag to stop the loop
  std::atomic_bool stoped_{false};

  // flag to indicate if the handler is running
  std::atomic_bool running_{false};
};

}  // namespace llm
