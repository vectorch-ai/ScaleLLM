#pragma once

#include <functional>
#include <memory>
#include <string>
#include <thread>

#include "engine/engine.h"
#include "request/output.h"
#include "request/status.h"
#include "sampling_params.h"
#include "scheduler/continuous_scheduler.h"

namespace llm {

using RequestCallback = std::function<bool(RequestOutput output)>;

struct ChatMessage {
  std::string role;
  std::string content;
};

// NOLINTNEXTLINE
class _LLMEngine {
 public:
  _LLMEngine(const std::string& model_path, const std::string& device_str);

  ~_LLMEngine();

  // schedule a request, the engine will execute the request asynchronously
  // and call the callback with output when the request is done
  // the callback will be called multiple times if the request is a streaming
  // request
  bool schedule_async(const std::string& prompt,
                      const SamplingParams& sp,
                      RequestCallback callback);

  // bool schedule_async(const std::vector<ChatMessage>& messages,
  //                     const SamplingParams& sp,
  //                     RequestCallback callback);

  // run the engine until stop() is called
  bool run_forever();

  // run the engine until no requests to process
  bool run_until_complete();

  // stop the engine
  void stop();

 private:
  std::unique_ptr<Engine> engine_;

  std::unique_ptr<ContinuousScheduler> scheduler_;

  // tokenizer instance
  std::unique_ptr<Tokenizer> tokenizer_;

  // model args
  ModelArgs model_args_;

  // thread pool for handling requests
  ThreadPool thread_pool_;

  std::thread loop_thread_;

  std::atomic_bool stoped_{false};
};

}  // namespace llm
