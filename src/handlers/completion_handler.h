#pragma once

#include <absl/container/flat_hash_set.h>

#include "call_data.h"
#include "completion.grpc.pb.h"  // IWYU pragma: keep
#include "handlers/llm_handler.h"

namespace llm {

using CompletionCallData =
    StreamCallData<proto::CompletionRequest, proto::CompletionResponse>;

// a class to handle completion requests
class CompletionHandler final {
 public:
  CompletionHandler(LLMHandler* llm_handler,
                    const std::vector<std::string>& models);

  // caller needs to guarantee the lifetime of call_data.
  void complete_async(CompletionCallData* call_data);

 private:
  // llm handler
  LLMHandler* llm_handler_;

  absl::flat_hash_set<std::string> models_;
};

}  // namespace llm
