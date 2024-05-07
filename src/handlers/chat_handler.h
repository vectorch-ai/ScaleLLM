#pragma once

#include <gflags/gflags.h>

#include "call_data.h"
#include "chat.grpc.pb.h"  // IWYU pragma: keep
#include "llm_handler.h"

namespace llm {
using ChatCallData = StreamCallData<proto::ChatRequest, proto::ChatResponse>;

// a class to handle completion requests
class ChatHandler final {
 public:
  ChatHandler(LLMHandler* llm_handler);

  // caller needs to guarantee the lifetime of call_data.
  void chat_async(ChatCallData* call_data);

 private:
  // request scheduler
  LLMHandler* llm_handler_;
};

}  // namespace llm
