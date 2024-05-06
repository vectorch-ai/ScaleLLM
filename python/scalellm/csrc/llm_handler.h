#pragma once

#include <functional>
#include <memory>
#include <string>
#include <thread>

#include "engine/engine.h"
#include "request/output.h"
#include "sampling_params.h"
#include "scheduler/continuous_scheduler.h"

namespace llm::csrc {

using RequestCallback = std::function<bool(RequestOutput output)>;

struct ChatMessage {
  std::string role;
  std::string content;
};

// NOLINTNEXTLINE
class LLMHandler {
 public:
  LLMHandler(const std::string& model_path, const std::string& device_str);

  ~LLMHandler();

  // schedule a request, the engine will execute the request asynchronously
  // and call the callback with output when the request is done
  // the callback will be called multiple times if the request is a streaming
  // request
  bool schedule(const std::string& prompt,
                const SamplingParams& sp,
                RequestCallback callback);

  // bool schedule_async(const std::vector<ChatMessage>& messages,
  //                     const SamplingParams& sp,
  //                     RequestCallback callback);

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

}  // namespace llm::csrc
