#pragma once

#include <glog/logging.h>
#include <grpcpp/grpcpp.h>

#include <string>
#include <thread>

#include "call_data.h"
#include "scheduler/scheduler.h"
#include "common/executor.h"

namespace llm {

// a class to handle completion requests
class CompletionHandler final {
 public:
  CompletionHandler(Scheduler* scheduler, const Tokenizer* tokenizer);

  ~CompletionHandler();

  // caller needs to guarantee the lifetime of call_data.
  void complete_async(CompletionCallData* call_data);

  // caller needs to guarantee the lifetime of call_data.
  void chat_async(ChatCallData* call_data);

 private:
  // request scheduler
  Scheduler* scheduler_;

  const Tokenizer* tokenizer_;

  // converter executor
  Executor converter_executor_;

  // scheduler loop thread
  std::thread scheduler_thread_;
};

}  // namespace llm
