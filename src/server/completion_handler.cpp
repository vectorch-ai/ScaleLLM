#include "completion_handler.h"

#include <glog/logging.h>
#include <grpcpp/grpcpp.h>
#include <torch/torch.h>

#include <cstdint>
#include <sstream>
#include <string>
#include <thread>

#include "call_data.h"
#include "completion.grpc.pb.h"
#include "models/args.h"
#include "request/request.h"

namespace llm {

namespace {

RequestPriority grpc_priority_to_priority(Priority priority) {
  switch (priority) {
    case Priority::DEFAULT:
      return RequestPriority::MEDIUM;
    case Priority::LOW:
      return RequestPriority::LOW;
    case Priority::MEDIUM:
      return RequestPriority::MEDIUM;
    case Priority::HIGH:
      return RequestPriority::HIGH;
    default:
      LOG(WARNING) << "Unknown priority: " << static_cast<int>(priority);
  }
  return RequestPriority::MEDIUM;
}

std::string finish_reason_to_string(FinishReason reason) {
  switch (reason) {
    case FinishReason::NONE:
      return "";
    case FinishReason::STOP:
      return "stop";
    case FinishReason::LENGTH:
      return "length";
    case FinishReason::FUNCTION_CALL:
      return "function_call";
    default:
      LOG(WARNING) << "Unknown finish reason: " << static_cast<int>(reason);
  }
  return "";
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
bool verify_request_arguments(CompletionCallData* call_data) {
  const auto& request = call_data->request();
  // n and best_of are not implemented yet
  if (request.has_n() && request.n() > 1) {
    call_data->finish_with_error(grpc::StatusCode::UNIMPLEMENTED,
                                 "n > 1 is not supported yet");
    return false;
  }
  if (request.has_best_of() && request.best_of() > 1) {
    call_data->finish_with_error(grpc::StatusCode::UNIMPLEMENTED,
                                 "best_of > 1 is not supported yet");
    return false;
  }

  // prompt is required
  if (request.prompt().empty()) {
    call_data->finish_with_error(grpc::StatusCode::INVALID_ARGUMENT,
                                 "Missing prompt");
    return false;
  }
  // temperature between [0.0, 2.0]
  if (request.has_temperature()) {
    if (request.temperature() < 0.0 || request.temperature() > 2.0) {
      call_data->finish_with_error(grpc::StatusCode::INVALID_ARGUMENT,
                                   "temperature must be between 0.0 and 2.0");
      return false;
    }
  }
  // top_p between [0.0, 1.0]
  if (request.has_top_p()) {
    if (request.top_p() < 0.0 || request.top_p() > 1.0) {
      call_data->finish_with_error(grpc::StatusCode::INVALID_ARGUMENT,
                                   "top_p must be between 0.0 and 1.0");
      return false;
    }
  }

  // logprobs <= 5
  if (request.has_logprobs()) {
    if (request.logprobs() > 5) {
      call_data->finish_with_error(grpc::StatusCode::INVALID_ARGUMENT,
                                   "logprobs must be between 0 and 5");
      return false;
    }
  }
  // presence_penalty between [-2.0, 2.0]
  if (request.has_presence_penalty()) {
    if (request.presence_penalty() < -2.0 || request.presence_penalty() > 2.0) {
      call_data->finish_with_error(
          grpc::StatusCode::INVALID_ARGUMENT,
          "presence_penalty must be between -2.0 and 2.0");
      return false;
    }
  }
  // frequency_penalty between [0.0, 2.0]
  if (request.has_frequency_penalty()) {
    if (request.frequency_penalty() < 0.0 ||
        request.frequency_penalty() > 2.0) {
      call_data->finish_with_error(
          grpc::StatusCode::INVALID_ARGUMENT,
          "frequency_penalty must be between 0.0 and 2.0");
      return false;
    }
  }
  // best_of >= n
  if (request.has_best_of() && request.has_n()) {
    if (request.best_of() < request.n()) {
      call_data->finish_with_error(grpc::StatusCode::INVALID_ARGUMENT,
                                   "best_of must be greater or equal to n");
      return false;
    }
  }
  return true;
}

std::unique_ptr<Request> grpc_request_to_request(CompletionCallData* call_data,
                                                 const Tokenizer& tokenizer,
                                                 const ModelArgs& model_args) {
  const CompletionRequest& grpc_request = call_data->request();
  CHECK(!grpc_request.prompt().empty()) << "Prompt is empty";

  const int64_t max_context_len = model_args.max_position_embeddings();

  std::vector<int> token_ids;
  if (!tokenizer.encode(grpc_request.prompt(), &token_ids)) {
    LOG(ERROR) << "Failed to encode prompt: " << grpc_request.prompt();
    return nullptr;
  }
  if (token_ids.size() > max_context_len) {
    LOG(ERROR) << "Prompt is too long: " << token_ids.size();
    return nullptr;
  }

  auto request = std::make_unique<Request>();

  // construct sampling parameters
  auto& sampling_param = request->sampling_param;
  if (grpc_request.has_frequency_penalty()) {
    sampling_param.frequency_penalty = grpc_request.frequency_penalty();
  }
  if (grpc_request.has_presence_penalty()) {
    sampling_param.presence_penalty = grpc_request.presence_penalty();
  }
  if (grpc_request.has_temperature()) {
    sampling_param.temperature = grpc_request.temperature();
  }
  if (grpc_request.has_top_p()) {
    sampling_param.top_p = grpc_request.top_p();
  }
  // TODO: add support for following extended parameters
  // sampling_param.repetition_penalty = grpc_request.repetition_penalty();
  // sampling_param.top_k = grpc_request.top_k();
  // sampling_param.do_sample = grpc_request.do_sample();
  // sampling_param.seed = grpc_request.seed();

  // construct stopping criteria
  auto& stopping_criteria = request->stopping_criteria;
  auto max_tokens = static_cast<uint32_t>(max_context_len - token_ids.size());
  if (grpc_request.has_max_tokens()) {
    max_tokens = std::min(max_tokens, grpc_request.max_tokens());
  } else {
    const uint32_t kDefaultMaxTokens = 128;
    max_tokens = std::min(max_tokens, kDefaultMaxTokens);
  }
  stopping_criteria.max_tokens = max_tokens;
  // stopping_criteria.ignore_eos_token = false;
  stopping_criteria.eos_token_id = model_args.eos_token_id();

  if (grpc_request.has_stream()) {
    request->stream = grpc_request.stream();
  }
  if (grpc_request.has_echo()) {
    request->echo = grpc_request.echo();
  }
  if (grpc_request.has_priority()) {
    request->priority = grpc_priority_to_priority(grpc_request.priority());
  }
  request->created_time = absl::ToUnixSeconds(absl::Now());

  // add on_stream and on_finish callbacks
  if (request->stream) {
    auto on_stream = [call_data, request = request.get()](
                         const std::string& delta,
                         FinishReason reason) -> bool {
      CompletionResponse response;
      response.set_object("text_completion");
      response.set_id(request->id);
      response.set_created(request->created_time);
      // response.set_model(request->model);
      auto* choice = response.add_choices();
      choice->set_text(delta);
      // choice->set_logprobs(0);
      choice->set_index(0);
      if (reason != FinishReason::NONE) {
        choice->set_finish_reason(finish_reason_to_string(reason));
      }
      return call_data->write(response);
    };

    request->add_sequence(
        grpc_request.prompt(), std::move(token_ids), on_stream);

    request->on_finish = [call_data, request = request.get()](
                             const std::string& output_text,
                             FinishReason reason,
                             const Status& status) -> bool {
      CHECK(output_text.empty());
      CHECK(reason == FinishReason::NONE);

      // TODO: mapping status to grpc status
      return call_data->finish();
    };
  } else {
    request->add_sequence(grpc_request.prompt(), std::move(token_ids), nullptr);
    request->on_finish = [call_data, request = request.get()](
                             const std::string& output_text,
                             FinishReason reason,
                             const Status& status) -> bool {
      CompletionResponse response;
      response.set_object("text_completion");
      response.set_id(request->id);
      response.set_created(request->created_time);
      // response.set_model(request->model);
      auto* choice = response.add_choices();
      choice->set_text(output_text);
      // choice->set_logprobs(0);
      choice->set_index(0);
      if (reason != FinishReason::NONE) {
        choice->set_finish_reason(finish_reason_to_string(reason));
      }
      // TODO: combine write and finish
      call_data->write(response);
      // TODO: mapping status to grpc status
      return call_data->finish();
    };
  }
  return request;
}

}  // namespace

CompletionHandler::CompletionHandler(Scheduler* scheduler, const Engine* engine)
    : scheduler_(scheduler) {
  CHECK(scheduler_ != nullptr);
  tokenizer_ = engine->tokenizer();
  model_args_ = engine->model_args();
}

CompletionHandler::~CompletionHandler() {}

void CompletionHandler::complete_async(CompletionCallData* call_data) {
  converter_executor_.schedule([this, call_data = call_data]() {
    if (!verify_request_arguments(call_data)) {
      // request is not valid, finish with error
      return;
    }

    auto request = grpc_request_to_request(call_data, *tokenizer_, model_args_);
    if (request == nullptr) {
      call_data->finish_with_error(grpc::StatusCode::UNKNOWN,
                                   "Unknown request processing error");
      return;
    }

    // schedule the request
    if (!scheduler_->schedule(request)) {
      call_data->finish_with_error(grpc::StatusCode::RESOURCE_EXHAUSTED,
                                   "Out of capacity");
    }
  });
}

}  // namespace llm
