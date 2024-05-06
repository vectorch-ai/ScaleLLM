#include "llm_handler.h"

#include <memory>
#include <thread>

#include "engine/engine_factory.h"
#include "request/output.h"
#include "request/request.h"

namespace llm {
namespace {
std::unique_ptr<Request> create_completion_request(const std::string& prompt,
                                                   const SamplingParams& sp,
                                                   RequestCallback callback) {
  // auto request = std::make_unique<Request>();
  // request->prompt = prompt;
  // request->sampling_params = sp;
  return nullptr;
}

std::unique_ptr<Request> create_chat_request(const std::string& prompt,
                                             const SamplingParams& sp,
                                             RequestCallback callback) {
  return nullptr;
}
}  // namespace

LLMHandler::LLMHandler(const std::string& model_path,
                       const std::string& device_str) {
  engine_ = EngineFactory::create(model_path, device_str);

  ContinuousScheduler::Options scheduler_options;
  scheduler_options.max_tokens_per_batch(512)
      .max_seqs_per_batch(128)
      .num_speculative_tokens(0);
  scheduler_ =
      std::make_unique<ContinuousScheduler>(engine_.get(), scheduler_options);

  loop_thread_ = std::thread([this]() {
    const auto timeout = absl::Milliseconds(500);
    while (!stoped_.load(std::memory_order_relaxed)) {
      // move scheduler forward
      scheduler_->step(timeout);
    }
  });
}

LLMHandler::~LLMHandler() {
  stop();
  loop_thread_.join();
}

bool LLMHandler::schedule(const std::string& prompt,
                          const SamplingParams& sp,
                          RequestCallback callback) {
  thread_pool_.schedule([this, prompt, sp, callback]() {
    // verify the prompt
    auto request = create_completion_request(prompt, sp, callback);
    if (!request) {
      // return error
    }
    if (!scheduler_->schedule(request)) {
      // TODO: return error
    }
  });
  return true;
}

// bool _LLMEngine::schedule_async(const std::vector<ChatMessage>& messages,
//                                 const SamplingParams& sp,
//                                 RequestCallback callback) {
//   return true;
// }

// stop the engine
void LLMHandler::stop() { stoped_.store(true, std::memory_order_relaxed); }

}  // namespace llm
