#include "chat_handler.h"

#include <absl/strings/escaping.h>
#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <glog/logging.h>
#include <grpcpp/grpcpp.h>
#include <torch/torch.h>

#include <boost/algorithm/string.hpp>
#include <cstdint>
#include <string>
#include <unordered_set>

#include "chat_template/chat_template.h"
#include "handlers/sampling_params.h"
#include "utils.h"
#include "uuid.h"

namespace llm {

namespace {
// NOLINTNEXTLINE
thread_local ShortUUID short_uuid;

std::string generate_request_id() { return "chatcmpl-" + short_uuid.random(); }

void set_logprobs(proto::ChatChoice* choice,
                  const std::optional<std::vector<LogProb>>& logprobs) {
  if (!logprobs.has_value() || logprobs.value().empty()) {
    return;
  }

  auto* proto_logprobs = choice->mutable_logprobs();
  for (const auto& logprob : logprobs.value()) {
    auto* logprob_proto = proto_logprobs->add_content();
    logprob_proto->set_token(logprob.token);
    logprob_proto->set_token_id(logprob.token_id);
    logprob_proto->set_logprob(logprob.logprob);

    if (logprob.top_logprobs.has_value()) {
      for (const auto& top_logprob : logprob.top_logprobs.value()) {
        auto* top_logprob_proto = logprob_proto->add_top_logprobs();
        top_logprob_proto->set_token(top_logprob.token);
        top_logprob_proto->set_token_id(top_logprob.token_id);
        top_logprob_proto->set_logprob(top_logprob.logprob);
      }
    }
  }
}

bool send_delta_to_client(ChatCallData* call_data,
                          bool include_usage,
                          std::unordered_set<size_t>* first_message_sent,
                          const std::string& request_id,
                          int64_t created_time,
                          const std::string& model,
                          const RequestOutput& output) {
  // send delta to client
  for (const auto& seq_output : output.outputs) {
    const auto& index = seq_output.index;

    // send first chunk with role as assistant
    if (first_message_sent->find(index) == first_message_sent->end()) {
      proto::ChatResponse response;
      response.set_object("chat.completion.chunk");
      response.set_id(request_id);
      response.set_created(created_time);
      response.set_model(model);
      auto* choice = response.add_choices();
      choice->set_index(index);
      auto* message = choice->mutable_delta();
      message->set_role("assistant");
      message->set_content("");
      // update first_message_sent
      first_message_sent->insert(index);
      if (!call_data->write(std::move(response))) {
        return false;
      }
    }

    // send chunk with delta message
    if (!seq_output.text.empty()) {
      proto::ChatResponse response;
      response.set_object("chat.completion.chunk");
      response.set_id(request_id);
      response.set_created(created_time);
      response.set_model(model);
      auto* choice = response.add_choices();
      choice->set_index(index);
      set_logprobs(choice, seq_output.logprobs);
      auto* message = choice->mutable_delta();
      message->set_content(seq_output.text);
      if (!call_data->write(std::move(response))) {
        return false;
      }
    }

    // send a separate chunk with finish reason
    if (seq_output.finish_reason.has_value()) {
      proto::ChatResponse response;
      response.set_object("chat.completion.chunk");
      response.set_id(request_id);
      response.set_created(created_time);
      response.set_model(model);
      auto* choice = response.add_choices();
      choice->set_index(index);
      choice->set_finish_reason(seq_output.finish_reason.value());
      if (!call_data->write(std::move(response))) {
        return false;
      }
    }
  }

  // send additional chunk for usage statistics
  if (include_usage && output.usage.has_value()) {
    const auto& usage = output.usage.value();
    proto::ChatResponse response;
    response.set_object("chat.completion.chunk");
    response.set_id(request_id);
    response.set_created(created_time);
    response.set_model(model);
    auto* proto_usage = response.mutable_usage();
    proto_usage->set_prompt_tokens(
        static_cast<int32_t>(usage.num_prompt_tokens));
    proto_usage->set_completion_tokens(
        static_cast<int32_t>(usage.num_generated_tokens));
    proto_usage->set_total_tokens(static_cast<int32_t>(usage.num_total_tokens));
    if (!call_data->write(std::move(response))) {
      return false;
    }
  }
  return true;
}

bool send_result_to_client(ChatCallData* call_data,
                           const std::string& request_id,
                           int64_t created_time,
                           const std::string& model,
                           const RequestOutput& req_output) {
  proto::ChatResponse response;
  response.set_object("chat.completion");
  response.set_id(request_id);
  response.set_created(created_time);
  response.set_model(model);

  for (const auto& output : req_output.outputs) {
    // add choices into response
    auto* choice = response.add_choices();
    choice->set_index(output.index);
    set_logprobs(choice, output.logprobs);
    auto* message = choice->mutable_message();
    message->set_role("assistant");
    message->set_content(output.text);
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
    const proto::ChatRequest& request) {
  SamplingParams sampling_params;
  if (request.has_max_tokens()) {
    sampling_params.max_tokens = request.max_tokens();
  }
  if (request.has_n()) {
    sampling_params.n = request.n();
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
    sampling_params.logprobs = request.logprobs();
  }
  if (request.has_top_logprobs()) {
    sampling_params.top_logprobs = request.top_logprobs();
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

ChatHandler::ChatHandler(LLMHandler* llm_handler,
                         const std::vector<std::string>& models)
    : llm_handler_(llm_handler), models_(models.begin(), models.end()) {
  CHECK(llm_handler != nullptr);
  CHECK(!models_.empty());
}

void ChatHandler::chat_async(ChatCallData* call_data) {
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

  std::vector<Message> messages;
  messages.reserve(grpc_request.messages_size());
  for (const auto& message : grpc_request.messages()) {
    messages.emplace_back(message.role(), message.content());
  }
  bool include_usage = false;
  if (grpc_request.has_stream_options()) {
    include_usage = grpc_request.stream_options().include_usage();
  }

  // schedule the request
  llm_handler_->schedule_chat_async(
      std::move(messages),
      std::move(sp),
      priority,
      stream,
      [call_data,
       model,
       stream = stream,
       include_usage = include_usage,
       first_message_sent = std::unordered_set<size_t>(),
       request_id = generate_request_id(),
       created_time = absl::ToUnixSeconds(absl::Now())](
          const RequestOutput& req_output) mutable -> bool {
        if (req_output.status.has_value()) {
          const auto& status = req_output.status.value();
          if (!status.ok()) {
            return call_data->finish_with_error(
                to_grpc_status_code(status.code()), status.message());
          }
        }

        if (stream) {
          // send delta to client
          return send_delta_to_client(call_data,
                                      include_usage,
                                      &first_message_sent,
                                      request_id,
                                      created_time,
                                      model,
                                      req_output);
        }
        return send_result_to_client(
            call_data, request_id, created_time, model, req_output);
      });
}

}  // namespace llm
