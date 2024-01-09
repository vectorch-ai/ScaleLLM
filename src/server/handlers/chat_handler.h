#pragma once

#include "server/call_data.h"
#include "chat.grpc.pb.h"  // IWYU pragma: keep
#include "common/threadpool.h"
#include "engine/engine.h"
#include "models/args.h"
#include "scheduler/scheduler.h"

namespace llm {
using ChatCallData = CallData<ChatRequest, ChatResponse>;

// a class to handle completion requests
class ChatHandler final {
 public:
  ChatHandler(Scheduler* scheduler, const Engine* engine);

  // caller needs to guarantee the lifetime of call_data.
  void chat_async(ChatCallData* call_data);

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
