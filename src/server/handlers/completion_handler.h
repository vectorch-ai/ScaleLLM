#pragma once

#include <string>
#include <thread>

#include "server/call_data.h"
#include "common/executor.h"
#include "completion.grpc.pb.h"
#include "engine/engine.h"
#include "models/args.h"
#include "scheduler/scheduler.h"

namespace llm {

using CompletionCallData = CallData<CompletionRequest, CompletionResponse>;

// a class to handle completion requests
class CompletionHandler final {
 public:
  CompletionHandler(Scheduler* scheduler, const Engine* engine);

  // caller needs to guarantee the lifetime of call_data.
  void complete_async(CompletionCallData* call_data);

 private:
  // request scheduler
  Scheduler* scheduler_;

  // tokenizer instance
  std::unique_ptr<Tokenizer> tokenizer_;

  // model args
  ModelArgs model_args_;

  // converter executor
  Executor converter_executor_;
};

}  // namespace llm
