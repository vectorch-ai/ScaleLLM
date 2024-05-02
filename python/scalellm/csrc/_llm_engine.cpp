#include "_llm_engine.h"

#include "request/request.h"

namespace llm {

LLMEngine_::LLMEngine_(const std::string& model_path,
                       const std::string& device_str) {}

void LLMEngine_::schedule_async(const std::string& prompt,
                                const SamplingParameter_& sp,
                                RequestCallback callback) {
  // TODO: Implement this function
}

// run the engine until stop() is called
bool LLMEngine_::run_forever() { return true; }

// stop the engine
void LLMEngine_::stop() {}

// run the engine until no requests to process
void LLMEngine_::run_until_complete() {}

}  // namespace llm
