#include "completion_handler.h"

#include <glog/logging.h>
#include <grpcpp/grpcpp.h>
#include <torch/torch.h>
#include <uuid.h>

#include <cstdint>
#include <string>

#include "engine/engine.h"
#include "models/model_args.h"
#include "request/output.h"
#include "request/request.h"
#include "scheduler/scheduler.h"
#include "utils.h"

DECLARE_int32(num_speculative_tokens);

namespace llm {

namespace {

std::string generate_request_id() {
  return "cmpl-" + uuids::to_string(uuids::uuid_system_generator{}());
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
bool verify_request_arguments(CompletionCallData* call_data) {
  const auto& request = call_data->request();
  const uint32_t n = request.has_n() ? request.n() : 1;
  if (request.has_best_of() && request.best_of() != n) {
    call_data->finish_with_error(grpc::StatusCode::UNIMPLEMENTED,
                                 "best_of != n is not supported yet");
    return false;
  }

  // prompt is required
  if (request.prompt().empty()) {
    call_data->finish_with_error(grpc::StatusCode::INVALID_ARGUMENT,
                                 "missing prompt");
    return false;
  }
  // up to 4 stop sequences
  if (request.stop_size() > 4) {
    call_data->finish_with_error(grpc::StatusCode::INVALID_ARGUMENT,
                                 "stop size is too large");
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

bool send_delta_to_client(CompletionCallData* call_data,
                          Request* request,
                          const SequenceOutput& seq_output) {
  if (!seq_output.text.empty()) {
    proto::CompletionResponse response;
    response.set_object("text_completion");
    response.set_id(request->id);
    response.set_created(request->created_time);
    // response.set_model(request->model);
    auto* choice = response.add_choices();
    choice->set_index(seq_output.index);
    choice->set_text(seq_output.text);
    if (!call_data->write(std::move(response))) {
      return false;
    }
  }

  if (seq_output.finish_reason.has_value()) {
    proto::CompletionResponse response;
    response.set_object("text_completion");
    response.set_id(request->id);
    response.set_created(request->created_time);
    // response.set_model(request->model);
    auto* choice = response.add_choices();
    choice->set_index(seq_output.index);
    choice->set_finish_reason(seq_output.finish_reason.value());
    if (!call_data->write(std::move(response))) {
      return false;
    }
  }
  return true;
}

bool send_result_to_client(CompletionCallData* call_data,
                           Request* request,
                           const Status& /*status*/,
                           const RequestOutput& req_output) {
  if (req_output.outputs.empty()) {
    // TODO: mapping status to grpc status
    return call_data->finish();
  }

  proto::CompletionResponse response;
  response.set_object("text_completion");
  response.set_id(request->id);
  response.set_created(request->created_time);
  // response.set_model(request->model);

  // add choices into response
  for (const auto& output : req_output.outputs) {
    auto* choice = response.add_choices();
    choice->set_index(output.index);
    choice->set_text(output.text);
    // choice->set_logprobs(0);
    if (output.finish_reason.has_value()) {
      choice->set_finish_reason(output.finish_reason.value());
    }
  }

  // add usage statistics
  if (req_output.usage.has_value()) {
    const auto& usage = req_output.usage.value();
    auto* proto_usage = response.mutable_usage();
    proto_usage->set_prompt_tokens(
        static_cast<int32_t>(usage.num_prompt_tokens));
    proto_usage->set_completion_tokens(
        static_cast<int32_t>(usage.num_generated_tokens));
    proto_usage->set_total_tokens(static_cast<int32_t>(usage.num_total_tokens));
  }
  // TODO: combine write and finish
  call_data->write(response);
  return call_data->finish();
}

std::unique_ptr<Request> grpc_request_to_request(CompletionCallData* call_data,
                                                 const Tokenizer& tokenizer,
                                                 const ModelArgs& model_args) {
  const auto& grpc_request = call_data->request();
  CHECK(!grpc_request.prompt().empty()) << "Prompt is empty";

  const int64_t max_context_len = model_args.max_position_embeddings();

  std::vector<int> prompt_tokens;
  if (!tokenizer.encode(grpc_request.prompt(), &prompt_tokens)) {
    call_data->finish_with_error(grpc::StatusCode::INVALID_ARGUMENT,
                                 "Failed to encode prompt");
    LOG(ERROR) << "Failed to encode prompt: " << grpc_request.prompt();
    return nullptr;
  }
  if (prompt_tokens.size() > max_context_len) {
    call_data->finish_with_error(grpc::StatusCode::INVALID_ARGUMENT,
                                 "Prompt is too long");
    LOG(ERROR) << "Prompt is too long: " << prompt_tokens.size();
    return nullptr;
  }

  uint32_t max_tokens = 0;
  if (grpc_request.has_max_tokens()) {
    max_tokens = grpc_request.max_tokens();
  } else {
    const uint32_t kDefaultMaxTokens = 16;
    max_tokens = kDefaultMaxTokens;
  }
  // allocate enough capacity for prompt tokens, max tokens, and speculative
  // tokens
  const size_t capacity = prompt_tokens.size() + max_tokens +
                          FLAGS_num_speculative_tokens + /*bouns_token*/ 1;

  const uint32_t num_seqs = grpc_request.has_n() ? grpc_request.n() : 1;
  auto request = std::make_unique<Request>(generate_request_id(),
                                           grpc_request.prompt(),
                                           prompt_tokens,
                                           capacity,
                                           num_seqs);

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
  stopping_criteria.max_tokens = max_tokens;
  stopping_criteria.max_context_len =
      max_context_len - FLAGS_num_speculative_tokens;
  // stopping_criteria.ignore_eos_token = false;
  stopping_criteria.eos_token_id = model_args.eos_token_id();

  // use stop token ids if specified in the request
  if (grpc_request.stop_token_ids_size() > 0) {
    const auto& stop_token_ids = grpc_request.stop_token_ids();
    stopping_criteria.stop_token_ids.insert(stop_token_ids.begin(),
                                            stop_token_ids.end());
  }

  for (const auto& stop_seq : grpc_request.stop()) {
    // encode stop sequence
    std::vector<int> token_ids;
    if (!tokenizer.encode(stop_seq, &token_ids)) {
      call_data->finish_with_error(grpc::StatusCode::INVALID_ARGUMENT,
                                   "Failed to encode stop sequence");
      LOG(ERROR) << "Failed to encode stop sequence: " << stop_seq;
      return nullptr;
    }
    stopping_criteria.stop_sequences.push_back(token_ids);
  }

  if (grpc_request.has_stream()) {
    request->stream = grpc_request.stream();
  }
  if (grpc_request.has_echo()) {
    request->echo = grpc_request.echo();
  }
  if (grpc_request.has_priority()) {
    request->priority = grpc_priority_to_priority(grpc_request.priority());
  }

  // set callbacks
  if (request->stream) {
    request->on_stream = [call_data, request = request.get()](
                             const RequestOutput& output) -> bool {
      for (const auto& output : output.outputs) {
        if (!send_delta_to_client(call_data, request, output)) {
          return false;
        }
      }
      return true;
    };
  }

  // add on_finish callback
  request->on_finish = [call_data, request = request.get()](
                           const Status& status,
                           const RequestOutput& req_output) -> bool {
    return send_result_to_client(call_data, request, status, req_output);
  };

  // add one sequence, rest will be added by scheduler
  request->add_sequence();
  return request;
}

}  // namespace

CompletionHandler::CompletionHandler(Scheduler* scheduler, const Engine* engine)
    : scheduler_(scheduler) {
  CHECK(scheduler_ != nullptr);
  tokenizer_ = engine->tokenizer();
  model_args_ = engine->model_args();
}

void CompletionHandler::complete_async(CompletionCallData* call_data) {
  converter_threadpool_.schedule([this, call_data = call_data]() {
    if (!verify_request_arguments(call_data)) {
      // request is not valid, finish with error
      return;
    }

    auto request = grpc_request_to_request(call_data, *tokenizer_, model_args_);
    if (request == nullptr) {
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
