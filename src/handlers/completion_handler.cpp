#include "completion_handler.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <glog/logging.h>
#include <grpcpp/grpcpp.h>
#include <torch/torch.h>

#include <cstdint>
#include <string>

#include "completion.pb.h"
#include "request/output.h"
#include "utils.h"
#include "uuid.h"

namespace llm {

namespace {
// NOLINTNEXTLINE
thread_local ShortUUID short_uuid;
std::string generate_request_id() { return "cmpl-" + short_uuid.random(); }

void set_logprobs(proto::Choice* choice,
                  const std::optional<std::vector<LogProb>>& logprobs) {
  if (!logprobs.has_value() || logprobs.value().empty()) {
    return;
  }

  auto* proto_logprobs = choice->mutable_logprobs();
  for (const auto& logprob : logprobs.value()) {
    proto_logprobs->add_tokens(logprob.token);
    proto_logprobs->add_token_ids(logprob.token_id);
    proto_logprobs->add_token_logprobs(logprob.logprob);
  }
}

bool send_delta_to_client(CompletionCallData* call_data,
                          const std::string& request_id,
                          int64_t created_time,
                          const std::string& model,
                          const RequestOutput& output) {
  for (const auto& seq_output : output.outputs) {
    if (!seq_output.text.empty()) {
      proto::CompletionResponse response;
      response.set_object("text_completion");
      response.set_id(request_id);
      response.set_created(created_time);
      response.set_model(model);
      auto* choice = response.add_choices();
      choice->set_index(seq_output.index);
      choice->set_text(seq_output.text);
      set_logprobs(choice, seq_output.logprobs);
      if (!call_data->write(std::move(response))) {
        return false;
      }
    }

    // send finish reason as a separate message
    if (seq_output.finish_reason.has_value()) {
      proto::CompletionResponse response;
      response.set_object("text_completion");
      response.set_id(request_id);
      response.set_created(created_time);
      response.set_model(model);
      auto* choice = response.add_choices();
      choice->set_index(seq_output.index);
      choice->set_text("");
      choice->set_finish_reason(seq_output.finish_reason.value());
      if (!call_data->write(std::move(response))) {
        return false;
      }
    }
  }
  return true;
}

bool send_result_to_client(CompletionCallData* call_data,
                           const std::string& request_id,
                           int64_t created_time,
                           const std::string& model,
                           const RequestOutput& req_output) {
  proto::CompletionResponse response;
  response.set_object("text_completion");
  response.set_id(request_id);
  response.set_created(created_time);
  response.set_model(model);

  // add choices into response
  for (const auto& output : req_output.outputs) {
    auto* choice = response.add_choices();
    choice->set_index(output.index);
    choice->set_text(output.text);
    set_logprobs(choice, output.logprobs);
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

  return call_data->write_and_finish(response);
}

SamplingParams grpc_request_to_sampling_params(
    const proto::CompletionRequest& request) {
  SamplingParams sampling_params;
  if (request.has_max_tokens()) {
    sampling_params.max_tokens = request.max_tokens();
  }
  if (request.has_n()) {
    sampling_params.n = request.n();
  }
  if (request.has_echo()) {
    sampling_params.echo = request.echo();
  }
  if (request.has_frequency_penalty()) {
    sampling_params.frequency_penalty = request.frequency_penalty();
  }
  if (request.has_presence_penalty()) {
    sampling_params.presence_penalty = request.presence_penalty();
  }
  if (request.has_repetition_penalty()) {
    sampling_params.repetition_penalty = request.repetition_penalty();
  }
  if (request.has_temperature()) {
    sampling_params.temperature = request.temperature();
  }
  if (request.has_top_p()) {
    sampling_params.top_p = request.top_p();
  }
  if (request.has_top_k()) {
    sampling_params.top_k = request.top_k();
  }
  if (request.has_logprobs()) {
    sampling_params.logprobs = true;
    sampling_params.top_logprobs = request.logprobs();
  }
  if (request.has_skip_special_tokens()) {
    sampling_params.skip_special_tokens = request.skip_special_tokens();
  }
  if (request.has_ignore_eos()) {
    sampling_params.ignore_eos = request.ignore_eos();
  }
  if (request.stop_size() > 0) {
    sampling_params.stop =
        std::vector<std::string>(request.stop().begin(), request.stop().end());
  }
  if (request.stop_token_ids_size() > 0) {
    sampling_params.stop_token_ids = std::vector<int32_t>(
        request.stop_token_ids().begin(), request.stop_token_ids().end());
  }
  return sampling_params;
}

}  // namespace

CompletionHandler::CompletionHandler(LLMHandler* llm_handler,
                                     const std::vector<std::string>& models)
    : llm_handler_(llm_handler), models_(models.begin(), models.end()) {
  CHECK(llm_handler_ != nullptr);
  CHECK(!models_.empty());
}

void CompletionHandler::complete_async(CompletionCallData* call_data) {
  const auto& grpc_request = call_data->request();
  // check if model is supported
  const auto& model = grpc_request.model();
  if (!models_.contains(model)) {
    call_data->finish_with_error(grpc::StatusCode::NOT_FOUND,
                                 "Model not supported");
    return;
  }

  auto sp = grpc_request_to_sampling_params(grpc_request);
  auto priority = to_priority(grpc_request.priority());
  auto stream = grpc_request.stream();

  // schedule the request
  llm_handler_->schedule_async(
      grpc_request.prompt(),
      sp = std::move(sp),
      priority,
      stream,
      [call_data,
       model,
       request_id = generate_request_id(),
       created_time = absl::ToUnixSeconds(absl::Now())](
          const RequestOutput& req_output) -> bool {
        if (req_output.status.has_value()) {
          const auto& status = req_output.status.value();
          if (!status.ok()) {
            return call_data->finish_with_error(
                to_grpc_status_code(status.code()), status.message());
          }
        }

        if (req_output.finished) {
          return send_result_to_client(
              call_data, request_id, created_time, model, req_output);
        }
        // send delta to client
        return send_delta_to_client(
            call_data, request_id, created_time, model, req_output);
      });
}

}  // namespace llm
