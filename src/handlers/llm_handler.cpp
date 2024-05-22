#include "llm_handler.h"

#include <glog/logging.h>

#include <atomic>
#include <memory>
#include <thread>
#include <utility>

#include "common/scope_guard.h"
#include "engine/utils.h"
#include "models/model_args.h"
#include "models/model_registry.h"
#include "request/output.h"
#include "request/request.h"
#include "speculative/speculative_engine.h"

namespace llm {
namespace {

#define CALLBACK_WITH_ERROR(CODE, MSG) callback(Status{CODE, MSG});

bool verify_params(const SamplingParams& sp, OutputCallback callback) {
  // up to 4 stop sequences
  if (sp.stop.has_value() && sp.stop.value().size() > 4) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT, "stop size is too large");
    return false;
  }

  // temperature between [0.0, 2.0]
  if (sp.temperature < 0.0 || sp.temperature > 2.0) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "temperature must be between 0.0 and 2.0");
    return false;
  }

  // top_p between [0.0, 1.0]
  if (sp.top_p < 0.0 || sp.top_p > 1.0) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "top_p must be between 0.0 and 1.0");
    return false;
  }

  // logprobs <= 5
  // if (sp.logprobs > 5) {
  //   CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
  //                                 "logprobs must be between 0 and 5");
  // }

  // presence_penalty between [-2.0, 2.0]
  if (sp.presence_penalty < -2.0 || sp.presence_penalty > 2.0) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "presence_penalty must be between -2.0 and 2.0");
    return false;
  }

  // frequency_penalty between [0.0, 2.0]
  if (sp.frequency_penalty < 0.0 || sp.frequency_penalty > 2.0) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "frequency_penalty must be between 0.0 and 2.0");
    return false;
  }
  return true;
}

}  // namespace

LLMHandler::LLMHandler(const Options& options) : options_(options) {
  // construct engine
  const auto devices = parse_devices(options.devices().value_or("auto"));
  LOG(INFO) << "Creating engine with devices: " << to_string(devices);

  // create a speculative engine if draft model path is provided
  const auto draft_model_path = options.draft_model_path().value_or("");
  if (!draft_model_path.empty()) {
    const auto draft_devices =
        parse_devices(options.draft_devices().value_or("auto"));
    LOG(INFO) << "Using draft devices: " << to_string(draft_devices);
    SpeculativeEngine::Options spec_options;
    spec_options.devices(devices)
        .draft_devices(draft_devices)
        .block_size(options.block_size())
        .max_cache_size(options.max_cache_size())
        .max_memory_utilization(options.max_memory_utilization())
        .enable_prefix_cache(options.enable_prefix_cache())
        .num_speculative_tokens(options.num_speculative_tokens())
        .enable_cuda_graph(options.enable_cuda_graph())
        .cuda_graph_max_seq_len(options.cuda_graph_max_seq_len())
        .cuda_graph_batch_sizes(options.cuda_graph_batch_sizes())
        .draft_cuda_graph_batch_sizes(options.draft_cuda_graph_batch_sizes());

    auto spec_engine = std::make_unique<SpeculativeEngine>(spec_options);
    CHECK(spec_engine->init(options.model_path(), draft_model_path));
    engine_ = std::move(spec_engine);
  } else {
    LLMEngine::Options eng_options;
    eng_options.devices(devices)
        .block_size(options.block_size())
        .max_cache_size(options.max_cache_size())
        .max_memory_utilization(options.max_memory_utilization())
        .enable_prefix_cache(options.enable_prefix_cache())
        .enable_cuda_graph(options.enable_cuda_graph())
        .cuda_graph_max_seq_len(options.cuda_graph_max_seq_len())
        .cuda_graph_batch_sizes(options.cuda_graph_batch_sizes());

    auto engine = std::make_unique<LLMEngine>(eng_options);
    CHECK(engine->init(options.model_path()));
    engine_ = std::move(engine);
  }

  model_args_ = engine_->model_args();

  ContinuousScheduler::Options scheduler_options;
  scheduler_options.max_tokens_per_batch(options.max_tokens_per_batch())
      .max_seqs_per_batch(options.max_seqs_per_batch())
      .num_speculative_tokens(options.num_speculative_tokens());
  scheduler_ =
      std::make_unique<ContinuousScheduler>(engine_.get(), scheduler_options);

  // construct chat template
  auto factory = ModelRegistry::get_default_chat_template_factory(
      model_args_.model_type());
  if (factory) {
    LOG(INFO) << "Using default chat template for model type: "
              << model_args_.model_type();
    chat_template_ = factory();
  } else {
    const auto& tokenizer_args = engine_->tokenizer_args();
    if (!tokenizer_args.chat_template().empty()) {
      LOG(WARNING) << "No default chat template found for model type: "
                   << model_args_.model_type();
    }
  }

  // construct tokenizers and handling threads
  const auto* tokenizer = engine_->tokenizer();
  for (size_t i = 0; i < options.num_handling_threads(); ++i) {
    // create a tokenizer for each thread for now
    tokenizers_.emplace_back(tokenizer->clone());
    handling_threads_.emplace_back([this, i] { handling_loop(i); });
  }
}

LLMHandler::~LLMHandler() {
  stop();

  // stop all handling threads
  // push nullptr to the queue to signal threads to exit
  for (size_t i = 0; i < handling_threads_.size(); ++i) {
    queue_.push(nullptr);
  }
  // wait for all threads to finish
  for (auto& thread : handling_threads_) {
    thread.join();
  }
}

void LLMHandler::schedule_async(std::string prompt,
                                SamplingParams sp,
                                Priority priority,
                                bool stream,
                                OutputCallback callback) {
  // add one pending request
  scheduler_->inc_pending_requests(1);
  schedule(
      std::move(prompt), std::move(sp), priority, stream, std::move(callback));
}

void LLMHandler::schedule_chat_async(std::vector<Message> messages,
                                     SamplingParams sp,
                                     Priority priority,
                                     bool stream,
                                     OutputCallback callback) {
  // add one pending request
  scheduler_->inc_pending_requests(1);
  schedule(std::move(messages),
           std::move(sp),
           priority,
           stream,
           std::move(callback));
}

void LLMHandler::schedule_batch_async(std::vector<std::string> prompts,
                                      std::vector<SamplingParams> sps,
                                      Priority priority,
                                      bool stream,
                                      BatchOutputCallback callback) {
  CHECK(prompts.size() == sps.size() || sps.size() == 1)
      << "Number of prompts and sampling parameters should be the same";

  const size_t num_requests = prompts.size();
  scheduler_->inc_pending_requests(num_requests);
  for (size_t i = 0; i < num_requests; ++i) {
    schedule(std::move(prompts[i]),
             // the sampling parameter may be shared
             sps.size() == 1 ? sps[0] : std::move(sps[i]),
             priority,
             stream,
             [i, callback](const RequestOutput& output) {
               return callback(i, output);
             });
  }
}

void LLMHandler::schedule_chat_batch_async(
    std::vector<std::vector<Message>> conversations,
    std::vector<SamplingParams> sps,
    Priority priority,
    bool stream,
    BatchOutputCallback callback) {
  CHECK(conversations.size() == sps.size() || sps.size() == 1)
      << "Number of conversations and sampling parameters should be the same";

  const size_t num_requests = conversations.size();
  scheduler_->inc_pending_requests(num_requests);
  for (size_t i = 0; i < conversations.size(); ++i) {
    schedule(std::move(conversations[i]),
             // the sampling parameter may be shared
             sps.size() == 1 ? sps[0] : std::move(sps[i]),
             priority,
             stream,
             [i, callback](const RequestOutput& output) {
               return callback(i, output);
             });
  }
}

void LLMHandler::schedule(std::string prompt,
                          SamplingParams sp,
                          Priority priority,
                          bool stream,
                          OutputCallback callback) {
  auto task = [this,
               prompt = std::move(prompt),
               sp = std::move(sp),
               priority,
               stream,
               callback](size_t tid) mutable {
    // remove the pending request after scheduling
    ScopeGuard cleanup = [this] { scheduler_->dec_pending_requests(); };

    // verify the prompt
    if (!verify_params(sp, callback)) {
      return;
    }

    auto request =
        create_request(tid, std::move(prompt), sp, priority, stream, callback);
    if (!request) {
      return;
    }

    if (!scheduler_->schedule(request)) {
      CALLBACK_WITH_ERROR(StatusCode::RESOURCE_EXHAUSTED,
                          "No available resources to schedule request");
      return;
    }
  };
  // add into the queue
  queue_.push(std::move(task));
}

void LLMHandler::schedule(std::vector<Message> messages,
                          SamplingParams sp,
                          Priority priority,
                          bool stream,
                          OutputCallback callback) {
  auto task = [this,
               messages = std::move(messages),
               sp = std::move(sp),
               priority,
               stream,
               callback](size_t tid) mutable {
    // remove the pending request after scheduling
    ScopeGuard cleanup = [this] { scheduler_->dec_pending_requests(); };
    // verify the prompt
    if (!verify_params(sp, callback)) {
      return;
    }

    auto request =
        create_chat_request(tid, messages, sp, priority, stream, callback);
    if (!request) {
      return;
    }

    if (!scheduler_->schedule(request)) {
      CALLBACK_WITH_ERROR(StatusCode::RESOURCE_EXHAUSTED,
                          "No available resources to schedule request");
      return;
    }
  };
  // add into the queue
  queue_.push(std::move(task));
}

void LLMHandler::handling_loop(size_t tid) {
  while (true) {
    Task task = queue_.pop();
    if (task == nullptr) {
      // nullptr is a signal to exit
      break;
    }
    task(tid);
  }
}

void LLMHandler::start() {
  loop_thread_ = std::thread([this]() {
    const bool running = running_.load(std::memory_order_relaxed);
    CHECK(!running) << "Handler is already running";

    running_.store(true, std::memory_order_relaxed);
    const auto timeout = absl::Milliseconds(500);
    while (!stoped_.load(std::memory_order_relaxed)) {
      // move scheduler forward
      scheduler_->step(timeout);
    }
    running_.store(false, std::memory_order_relaxed);
  });
}

// stop the engine
void LLMHandler::stop() {
  // set stop flag
  stoped_.store(true, std::memory_order_relaxed);
  // wait for the loop thread to finish
  if (loop_thread_.joinable()) {
    loop_thread_.join();
  }
}

void LLMHandler::run_until_complete() {
  const bool running = running_.load(std::memory_order_relaxed);
  CHECK(!running) << "Handler is already running";

  running_.store(true, std::memory_order_relaxed);
  scheduler_->run_until_complete();
  running_.store(false, std::memory_order_relaxed);
}

std::unique_ptr<Request> LLMHandler::create_request(size_t tid,
                                                    std::string prompt,
                                                    const SamplingParams& sp,
                                                    Priority priority,
                                                    bool stream,
                                                    OutputCallback callback) {
  CHECK(!prompt.empty()) << "Prompt should not be empty";

  // encode the prompt
  std::vector<int> prompt_tokens;
  if (!tokenizers_[tid]->encode(prompt, &prompt_tokens)) {
    LOG(ERROR) << "Failed to encode prompt: " << prompt;
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "Failed to encode prompt");
    return nullptr;
  }

  const int64_t max_context_len = model_args_.max_position_embeddings();
  if (prompt_tokens.size() >= max_context_len) {
    LOG(ERROR) << "Prompt is too long: " << prompt_tokens.size();
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT, "Prompt is too long");
    return nullptr;
  }

  uint32_t max_tokens = sp.max_tokens;
  if (max_tokens == 0) {
    const uint32_t kDefaultMaxTokens = 16;
    max_tokens = kDefaultMaxTokens;
  }

  const uint32_t num_seqs = std::max<uint32_t>(1, sp.n);
  // allocate enough capacity for prompt tokens, max tokens, and speculative
  // tokens
  const size_t capacity = prompt_tokens.size() + max_tokens +
                          options_.num_speculative_tokens() + /*bouns_token*/ 1;
  auto request = std::make_unique<Request>(
      std::move(prompt), std::move(prompt_tokens), capacity, num_seqs);

  // sampling parameters
  auto& sampling_param = request->sampling_param;
  sampling_param.frequency_penalty = sp.frequency_penalty;
  sampling_param.presence_penalty = sp.presence_penalty;
  sampling_param.repetition_penalty = sp.repetition_penalty;
  sampling_param.temperature = sp.temperature;
  sampling_param.top_p = sp.top_p;
  sampling_param.top_k = sp.top_k;
  // sampling_param.do_sample = sp.do_sample;
  // sampling_param.seed = sp.seed;

  // stopping criteria
  auto& stopping_criteria = request->stopping_criteria;
  stopping_criteria.max_tokens = max_tokens;
  stopping_criteria.max_context_len =
      max_context_len - options_.num_speculative_tokens();
  stopping_criteria.ignore_eos = sp.ignore_eos;
  stopping_criteria.eos_token_id = model_args_.eos_token_id();

  if (sp.stop_token_ids.has_value()) {
    const auto& stop_token_ids = sp.stop_token_ids.value();
    stopping_criteria.stop_token_ids.insert(stop_token_ids.begin(),
                                            stop_token_ids.end());
  } else {
    // otherwise use default stop token id from model args
    stopping_criteria.stop_token_ids = model_args_.stop_token_ids();
  }

  if (sp.stop.has_value()) {
    for (const auto& s : sp.stop.value()) {
      std::vector<int> stop_tokens;
      if (!tokenizers_[tid]->encode(s, &stop_tokens)) {
        CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                            "Failed to encode stop sequence");
        LOG(ERROR) << "Failed to encode stop sequence: " << s;
        return nullptr;
      }
      stopping_criteria.stop_sequences.push_back(std::move(stop_tokens));
    }
  }
  request->stream = stream;
  request->priority = priority;
  request->echo = sp.echo;

  // set callback for outputs
  request->on_output = callback;

  // add one sequence, rest will be added by scheduler
  request->add_sequence();
  return request;
}

std::unique_ptr<Request> LLMHandler::create_chat_request(
    size_t tid,
    const std::vector<Message>& messages,
    const SamplingParams& sp,
    Priority priority,
    bool stream,
    OutputCallback callback) {
  // construct prompt from dialog messages
  if (chat_template_ == nullptr) {
    CALLBACK_WITH_ERROR(
        StatusCode::INVALID_ARGUMENT,
        "Chat template has not configured, please use /completion API");
    LOG(ERROR) << "Chat template has not configured for model type: "
               << model_args_.model_type();
    return nullptr;
  }

  auto prompt = chat_template_->apply(messages);
  if (!prompt.has_value()) {
    CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                        "Failed to construct prompt from messages");
    LOG(ERROR) << "Failed to construct prompt from messages";
    return nullptr;
  }

  return create_request(
      tid, std::move(prompt.value()), sp, priority, stream, callback);
}

}  // namespace llm
