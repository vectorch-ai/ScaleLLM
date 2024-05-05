#pragma once

#include <functional>
#include <string>

#include "request/output.h"
#include "sampling_params.h"

namespace llm {

using RequestCallback = std::function<bool(const RequestOutput& output)>;
using ChatMessage = std::pair<std::string /*role*/, std::string /*content*/>;

// NOLINTNEXTLINE
class _LLMEngine {
 public:
  _LLMEngine(const std::string& model_path, const std::string& device_str);

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
};

}  // namespace llm
