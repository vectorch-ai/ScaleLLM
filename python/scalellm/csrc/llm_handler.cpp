#include "llm_handler.h"

#include <memory>
#include <thread>

#include "engine/engine_factory.h"
#include "request/output.h"
#include "request/request.h"

DECLARE_int32(num_speculative_tokens);

namespace llm::csrc {
namespace {
std::unique_ptr<Request> create_completion_request(
    const std::string& prompt,
    const SamplingParams& sp,
    RequestCallback callback,
    const Tokenizer& tokenizer,
    const ModelArgs& model_args) {
  CHECK(!prompt.empty()) << "Prompt should not be empty";

  const int64_t max_context_len = model_args.max_position_embeddings();

  // encode the prompt
  std::vector<int> prompt_tokens;
  if (!tokenizer.encode(prompt, &prompt_tokens)) {
    LOG(ERROR) << "Failed to encode prompt: " << prompt;
    return nullptr;
  }

  if (prompt_tokens.size() >= max_context_len) {
    LOG(ERROR) << "Prompt is too long: " << prompt_tokens.size();
    return nullptr;
  }

  uint32_t max_tokens = sp.max_tokens;
  if (max_tokens == 0) {
    const uint32_t kDefaultMaxTokens = 16;
    max_tokens = kDefaultMaxTokens;
  }

  // allocate enough capacity for prompt tokens, max tokens, and speculative
  // tokens
  const size_t capacity = prompt_tokens.size() + max_tokens +
                          FLAGS_num_speculative_tokens + /*bouns_token*/ 1;
  const uint32_t num_seqs = std::max<uint32_t>(1, sp.n);
  auto request = std::make_unique<Request>(
      "request_id", prompt, prompt_tokens, capacity, num_seqs);

  return request;
}

// std::unique_ptr<Request> create_chat_request(const std::string& prompt,
//                                              const SamplingParams& sp,
//                                              RequestCallback callback) {
//   return nullptr;
// }
}  // namespace

LLMHandler::LLMHandler(const std::string& model_path,
                       const std::string& device_str) {
  engine_ = EngineFactory::create(model_path, device_str);

  tokenizer_ = engine_->tokenizer();
  model_args_ = engine_->model_args();

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

LLMHandler::~LLMHandler() { stop(); }

bool LLMHandler::schedule(const std::string& prompt,
                          const SamplingParams& sp,
                          RequestCallback callback) {
  thread_pool_.schedule([this, prompt, sp, callback]() {
    // verify the prompt
    auto request = create_completion_request(
        prompt, sp, callback, *tokenizer_, model_args_);
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
void LLMHandler::stop() {
  // set stop flag
  stoped_.store(true, std::memory_order_relaxed);
  // wait for the loop thread to finish
  if (loop_thread_.joinable()) {
    loop_thread_.join();
  }
}

}  // namespace llm::csrc
