#pragma once

#include <functional>
#include <string>

#include "_sampling_parameter.h"
#include "request/output.h"

namespace llm {

using RequestCallback = std::function<void(const RequestOutput& output)>;

class LLMEngine_ {
 public:
  LLMEngine_(const std::string& model_path, const std::string& device_str);

  // schedule a request, the engine will execute the request asynchronously
  // and call the callback with output when the request is done
  // the callback will be called multiple times if the request is a streaming
  // request
  void schedule_async(const std::string& prompt,
                      const SamplingParameter_& sp,
                      RequestCallback callback);

  // run the engine until stop() is called
  bool run_forever();

  // stop the engine
  void stop();
  
  // run the engine until no requests to process
  void run_until_complete();

 private:
};

}  // namespace llm
