#include "chat_handler.h"

#include <absl/strings/escaping.h>
#include <glog/logging.h>
#include <grpcpp/grpcpp.h>
#include <torch/torch.h>
#include <uuid.h>

#include <boost/algorithm/string.hpp>
#include <cstdint>
#include <string>
#include <unordered_set>

#include "chat_template/chat_template.h"
#include "chat_template/jinja_chat_template.h"
#include "engine/engine.h"
#include "handlers/sampling_params.h"
#include "models/model_args.h"
#include "models/model_registry.h"
#include "request/request.h"
#include "scheduler/scheduler.h"
#include "utils.h"

DEFINE_bool(enable_jinja_chat_template, false, "Enable Jinja chat template");

DECLARE_int32(num_speculative_tokens);

namespace llm {

namespace {

std::string generate_request_id() {
  return "chatcmpl-" + uuids::to_string(uuids::uuid_system_generator{}());
}

bool send_delta_to_client(ChatCallData* call_data,
                          std::unordered_set<size_t>* first_message_sent,
                          const RequestOutput& output) {
  // send delta to client
  for (const auto& seq_output : output.outputs) {
    proto::ChatResponse response;
    response.set_object("chat.completion.chunk");
    // response.set_id(request->id);
    // response.set_created(request->created_time);
    // response.set_model(request->model);
    auto* choice = response.add_choices();
    const auto& index = seq_output.index;
    choice->set_index(index);
    // add message
    auto* message = choice->mutable_delta();
    // only set role for first message
    if (first_message_sent->find(index) == first_message_sent->end()) {
      message->set_role("assistant");
      first_message_sent->insert(index);
    }
    message->set_content(seq_output.text);
    if (seq_output.finish_reason.has_value()) {
      choice->set_finish_reason(seq_output.finish_reason.value());
    }
    if (!call_data->write(std::move(response))) {
      return false;
    }
  }
  return true;
}

bool send_result_to_client(ChatCallData* call_data,
                           const RequestOutput& req_output) {
  if (req_output.outputs.empty()) {
    // TODO: mapping status to grpc status
    return call_data->finish();
  }

  proto::ChatResponse response;
  response.set_object("chat.completion");
  // response.set_id(request->id);
  // response.set_created(request->created_time);
  // response.set_model(request->model);

  for (const auto& output : req_output.outputs) {
    // add choices into response
    auto* choice = response.add_choices();
    choice->set_index(output.index);
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

  // TODO: combine write and finish
  call_data->write(response);
  return call_data->finish();
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

ChatHandler::ChatHandler(LLMHandler* llm_handler) : llm_handler_(llm_handler) {
  CHECK(llm_handler != nullptr);
}

void ChatHandler::chat_async(ChatCallData* call_data) {
  const auto& grpc_request = call_data->request();
  auto sp = grpc_request_to_sampling_params(grpc_request);
  auto priority = grpc_priority_to_priority(grpc_request.priority());
  auto stream = grpc_request.stream();

  std::vector<Message> messages;
  messages.reserve(grpc_request.messages_size());
  for (const auto& message : grpc_request.messages()) {
    messages.emplace_back(message.role(), message.content());
  }

  // schedule the request
  llm_handler_->schedule_chat_async(
      std::move(messages),
      std::move(sp),
      priority,
      stream,
      [call_data, first_message_sent = std::unordered_set<size_t>()](
          const RequestOutput& req_output) mutable -> bool {
        if (req_output.finished) {
          return send_result_to_client(call_data, req_output);
        }
        // send delta to client
        return send_delta_to_client(call_data, &first_message_sent, req_output);
      });
}

}  // namespace llm
