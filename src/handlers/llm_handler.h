#pragma once

#include <functional>
#include <future>
#include <memory>
#include <string>
#include <thread>

#include "chat_template/chat_template.h"
#include "engine/engine.h"
#include "request/output.h"
#include "sampling_params.h"
#include "scheduler/continuous_scheduler.h"

namespace llm {

using OutputCallback = std::function<bool(RequestOutput output)>;

class ScheduleTask {
 public:
  ScheduleTask(std::future<bool> finish) : finish_(std::move(finish)) {}

  // wait until the request has been scheduled
  void wait() { finish_.wait(); }

  // get the status of the request scheduling
  bool get() { return finish_.get(); }

 private:
  std::future<bool> finish_;
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

  // start the handling loop
  void start();

  // stop the engine
  void stop();

  // run until complete, blocking call
  void run_until_complete();

 private:
  std::unique_ptr<Engine> engine_;

  std::unique_ptr<Scheduler> scheduler_;

  // tokenizer instance
  std::unique_ptr<Tokenizer> tokenizer_;

  // model args
  ModelArgs model_args_;

  // thread pool for handling requests
  ThreadPool thread_pool_;

  // chat template instance
  std::unique_ptr<ChatTemplate> chat_template_;

  std::thread loop_thread_;

  std::atomic_bool stoped_{false};
};

}  // namespace llm
