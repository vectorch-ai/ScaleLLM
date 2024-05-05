#include "_llm_engine.h"

#include <thread>

#include "request/request.h"

namespace llm {

_LLMEngine::_LLMEngine(const std::string& model_path,
                       const std::string& device_str) {}

bool _LLMEngine::schedule_async(const std::string& prompt,
                                const SamplingParams& sp,
                                RequestCallback callback) {
  // dummy implementation for testing
  std::thread([prompt, sp, callback]() mutable {
    for (size_t i = 0; i < 10; ++i) {
      RequestOutput output;
      output.finished = (i == 9);
      if (output.finished) {
        output.stats.num_prompt_tokens = 10;
        output.stats.num_generated_tokens = 20;
        output.stats.num_total_tokens = 30;
      }

      output.outputs.emplace_back();
      output.outputs.back().index = 0;
      output.outputs.back().text = std::to_string(i);
      callback(output);
    }
  }).detach();
  return true;
}

// bool _LLMEngine::schedule_async(const std::vector<ChatMessage>& messages,
//                                 const SamplingParams& sp,
//                                 RequestCallback callback) {
//   return true;
// }

// run the engine until stop() is called
bool _LLMEngine::run_forever() { return true; }

// run the engine until no requests to process
bool _LLMEngine::run_until_complete() { return true; }

// stop the engine
void _LLMEngine::stop() {}

}  // namespace llm
