#pragma once

#include "call_data.h"
#include "common/threadpool.h"
#include "completion.grpc.pb.h"  // IWYU pragma: keep
#include "models/model_args.h"
#include "tokenizer/tokenizer.h"

namespace llm {

using CompletionCallData =
    StreamCallData<CompletionRequest, CompletionResponse>;

class Scheduler;
class Engine;

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
