#include "completion_handler.h"

#include <glog/logging.h>
#include <grpcpp/grpcpp.h>
#include <torch/torch.h>
#include <uuid.h>

#include <cstdint>
#include <string>

#include "models/model_args.h"
#include "request/request.h"
#include "utils.h"

namespace llm {

namespace {

std::string generate_request_id() {
  return "cmpl-" + uuids::to_string(uuids::uuid_system_generator{}());
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
bool verify_request_arguments(CompletionCallData* call_data) {
  const auto& request = call_data->request();
  // n is not implemented yet for stream request
  const bool stream = request.has_stream() ? request.stream() : false;
  const uint32_t n = request.has_n() ? request.n() : 1;
  if (stream && n > 1) {
    call_data->finish_with_error(grpc::StatusCode::UNIMPLEMENTED,
                                 "n > 1 is not supported yet");
    return false;
  }

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
                          uint32_t index,
                          const std::string& delta,
                          FinishReason reason) {
  CompletionResponse response;
  response.set_object("text_completion");
  response.set_id(request->id);
  response.set_created(request->created_time);
  // response.set_model(request->model);
  auto* choice = response.add_choices();
  choice->set_index(index);
  choice->set_text(delta);
  if (reason != FinishReason::NONE) {
    choice->set_finish_reason(finish_reason_to_string(reason));
  }
  return call_data->write(response);
}

bool send_result_to_client(CompletionCallData* call_data,
                           Request* request,
                           const std::vector<SequenceResult>& seq_results,
                           const Status& /*status*/,
                           const Statistics& stats) {
  CompletionResponse response;
  response.set_object("text_completion");
  response.set_id(request->id);
  response.set_created(request->created_time);
  // response.set_model(request->model);

  // add choices into response
  for (uint32_t i = 0; i < seq_results.size(); ++i) {
    const auto& seq_result = seq_results[i];
    auto* choice = response.add_choices();
    choice->set_index(i);
    choice->set_text(seq_result.output_text);
    // choice->set_logprobs(0);
    if (seq_result.finish_reason != FinishReason::NONE) {
      choice->set_finish_reason(
          finish_reason_to_string(seq_result.finish_reason));
    }
  }

  // add usage statistics
  auto* usage = response.mutable_usage();
  usage->set_prompt_tokens(static_cast<int32_t>(stats.num_prompt_tokens));
  usage->set_completion_tokens(
      static_cast<int32_t>(stats.num_generated_tokens));
  usage->set_total_tokens(static_cast<int32_t>(stats.num_total_tokens));
  // TODO: combine write and finish
  call_data->write(response);
  // TODO: mapping status to grpc status
  return call_data->finish();
}

std::unique_ptr<Request> grpc_request_to_request(CompletionCallData* call_data,
                                                 const Tokenizer& tokenizer,
                                                 const ModelArgs& model_args) {
  const CompletionRequest& grpc_request = call_data->request();
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

  auto request = std::make_unique<Request>(
      generate_request_id(), grpc_request.prompt(), prompt_tokens);

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
  auto max_tokens =
      static_cast<uint32_t>(max_context_len - prompt_tokens.size());
  if (grpc_request.has_max_tokens()) {
    max_tokens = std::min(max_tokens, grpc_request.max_tokens());
  } else {
    const uint32_t kDefaultMaxTokens = 128;
    max_tokens = std::min(max_tokens, kDefaultMaxTokens);
  }
  stopping_criteria.max_tokens = max_tokens;
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

  // add on_stream and on_finish callbacks
  const uint32_t num_seqs = grpc_request.has_n() ? grpc_request.n() : 1;
  if (request->stream) {
    // add sequences with on_stream callback
    for (uint32_t i = 0; i < num_seqs; ++i) {
      request->add_sequence(
          [call_data, request = request.get(), i](const std::string& delta,
                                                  FinishReason reason) -> bool {
            return send_delta_to_client(call_data, request, i, delta, reason);
          });
    }

    // add on_stream_finish callback
    request->on_stream_finish = [call_data](const Status& /*status*/) -> bool {
      return call_data->finish();
    };
  } else {
    // add sequences
    for (uint32_t i = 0; i < num_seqs; ++i) {
      request->add_sequence();
    }

    // add on_finish callback
    request->on_finish = [call_data, request = request.get()](
                             const std::vector<SequenceResult>& seq_results,
                             const Status& status,
                             const Statistics& stats) -> bool {
      return send_result_to_client(
          call_data, request, seq_results, status, stats);
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
