#pragma once

#include "call_data.h"
#include "common/threadpool.h"
#include "completion.grpc.pb.h"  // IWYU pragma: keep
#include "engine/engine.h"
#include "models/model_args.h"
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

  // converter threadpool
  ThreadPool converter_threadpool_;
};

}  // namespace llm
