#include "completion_handler.h"

#include <glog/logging.h>
#include <grpcpp/grpcpp.h>
#include <torch/torch.h>
#include <uuid.h>

#include <cstdint>
#include <string>

#include "request/output.h"
#include "utils.h"

namespace llm {

namespace {

std::string generate_request_id() {
  return "cmpl-" + uuids::to_string(uuids::uuid_system_generator{}());
}

bool send_delta_to_client(CompletionCallData* call_data,
                          const RequestOutput& output) {
  for (const auto& seq_output : output.outputs) {
    if (!seq_output.text.empty()) {
      proto::CompletionResponse response;
      response.set_object("text_completion");
      // response.set_id(request->id);
      // response.set_created(request->created_time);
      // response.set_model(request->model);
      auto* choice = response.add_choices();
      choice->set_index(seq_output.index);
      choice->set_text(seq_output.text);
      if (seq_output.finish_reason.has_value()) {
        choice->set_finish_reason(seq_output.finish_reason.value());
      }
      if (!call_data->write(std::move(response))) {
        return false;
      }
    }
  }
  return true;
}

bool send_result_to_client(CompletionCallData* call_data,
                           const RequestOutput& req_output) {
  if (req_output.outputs.empty()) {
    // TODO: mapping status to grpc status
    return call_data->finish();
  }

  proto::CompletionResponse response;
  response.set_object("text_completion");
  // response.set_id(request->id);
  // response.set_created(request->created_time);
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
  // if (request.has_repetition_penalty()) {
  //   sampling_params.repetition_penalty = request.repetition_penalty();
  // }
  if (request.has_temperature()) {
    sampling_params.temperature = request.temperature();
  }
  if (request.has_top_p()) {
    sampling_params.top_p = request.top_p();
  }
  // if (request.has_top_k()) {
  //   sampling_params.top_k = request.top_k();
  // }
  // if (request.has_skip_special_tokens()) {
  //   sampling_params.skip_special_tokens = request.skip_special_tokens();
  // }
  // if (request.has_ignore_eos()) {
  //   sampling_params.ignore_eos = request.ignore_eos();
  // }
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

CompletionHandler::CompletionHandler(LLMHandler* llm_handler)
    : llm_handler_(llm_handler) {
  CHECK(llm_handler_ != nullptr);
}

void CompletionHandler::complete_async(CompletionCallData* call_data) {
  const auto& grpc_request = call_data->request();
  auto sp = grpc_request_to_sampling_params(grpc_request);
  auto priority = grpc_priority_to_priority(grpc_request.priority());
  auto stream = grpc_request.stream();

  // schedule the request
  llm_handler_->schedule_async(
      grpc_request.prompt(),
      sp = std::move(sp),
      priority,
      stream,
      [call_data](const RequestOutput& req_output) -> bool {
        if (req_output.finished) {
          return send_result_to_client(call_data, req_output);
        }
        // send delta to client
        return send_delta_to_client(call_data, req_output);
      });
}

}  // namespace llm
