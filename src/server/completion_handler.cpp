#include "completion_handler.h"

#include <absl/time/time.h>
#include <glog/logging.h>
#include <grpcpp/grpcpp.h>
#include <torch/torch.h>

#include <string>
#include <thread>

#include "call_data.h"
#include "completion.grpc.pb.h"
#include "request/request.h"
#include "model_loader/model_loader.h"

constexpr int kStepTimeoutMs = 500;

DEFINE_int32(num_converter_threads, 1, "number of converter threads");

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

std::unique_ptr<Request> grpc_completion_request_to_request(
    CompletionCallData* call_data,
    const Tokenizer* tokenizer) {
  const CompletionRequest& grpc_request = call_data->request();
  std::vector<int> token_ids;
  // token_ids.reserve(max_context_len);
  if (!tokenizer->encode(grpc_request.prompt(), &token_ids)) {
    LOG(ERROR) << "Failed to encode prompt: " << grpc_request.prompt();
    return nullptr;
  }

  // std::string prompt,
  // std::vector<int32_t> token_ids,
  // const SamplingParameter* sampling_param,
  // const StoppingCriteria* stopping_criteria

  auto request = std::make_unique<Request>();
  // TODO: generate unique id
  request->id = "unique id";

  // construct sampling parameters
  auto& sampling_param = request->sampling_param;
  sampling_param.frequency_penalty = grpc_request.frequency_penalty();
  sampling_param.presence_penalty = grpc_request.presence_penalty();
  // sampling_param.repetition_penalty = grpc_request.repetition_penalty();
  sampling_param.temperature = grpc_request.temperature();
  sampling_param.top_p = grpc_request.top_p();
  // sampling_param.top_k = grpc_request.top_k();
  // sampling_param.do_sample = grpc_request.do_sample();
  // sampling_param.seed = grpc_request.seed();

  // construct stopping criteria
  auto& stopping_criteria = request->stopping_criteria;
  // TODO: add better protection
  auto max_tokens = static_cast<uint32_t>(FLAGS_max_position_embeddings - token_ids.size());
  if (grpc_request.max_tokens() != 0) {
    max_tokens = std::min(max_tokens, grpc_request.max_tokens());
  }
  stopping_criteria.max_tokens = max_tokens;

  // stopping_criteria.ignore_eos_token = grpc_request.ignore_eos_token();
  // stopping_criteria.eos_token_id = tokenizer->eos_id();

  request->stream = grpc_request.stream();
  request->priority = grpc_priority_to_priority(grpc_request.priority());
  request->created_time = absl::ToUnixMicros(absl::Now());

  // TODO: handle best_of and n
  request->add_sequence(
      grpc_request.prompt(),
      std::move(token_ids),
      [call_data, request = request.get()](const std::string& delta,
                                           const FinishReason& reason) {
        CompletionResponse response;
        response.set_object("text_completion");
        response.set_id(request->id);
        response.set_created(request->created_time);
        // response.set_model(request->model);
        auto* choice = response.add_choices();
        choice->set_text(delta);
        // choice->set_logprobs(0);
        choice->set_index(0);
        // choice->set_finish_reason(static_cast<int>(reason));
        call_data->write(response);
      });

  request->on_finish = [call_data, request = request.get()](
                           const std::string& output_text,
                           const Status& status) {
    // TODO: handle best_of and n
    // TODO: mapping status to grpc status
    call_data->finish();
  };
  return request;
}

std::unique_ptr<Request> grpc_chat_request_to_request(
    ChatCallData* call_data,
    const Tokenizer* tokenizer) {
  auto request = std::make_unique<Request>();
  return request;
}

}  // namespace

CompletionHandler::CompletionHandler(Scheduler* scheduler,
                                     std::unique_ptr<Tokenizer> tokenizer)
    : scheduler_(scheduler),
      tokenizer_(std::move(tokenizer)),
      converter_executor_(FLAGS_num_converter_threads) {
  CHECK(scheduler_ != nullptr);
  CHECK(tokenizer_ != nullptr);
  // start the scheduler loop
  scheduler_thread_ = std::thread([this]() {
    torch::InferenceMode guard;
    const auto timeout = absl::Milliseconds(kStepTimeoutMs);
    while (true) {
      scheduler_->step(timeout);
    }
  });
}

CompletionHandler::~CompletionHandler() {
  // TODO: stop the scheduler loop
  // scheduler_thread_.join();
}

void CompletionHandler::complete_async(CompletionCallData* call_data) {
  converter_executor_.schedule([this, call_data = call_data]() {
    auto request = grpc_completion_request_to_request(call_data, tokenizer_.get());
    if (request == nullptr) {
      // TODO: finish with error
    }

    bool success = scheduler_->schedule(request);
    if (!success) {
      // TODO: finish with error: out of capacity
      // call_data->finish();
      // request->finish();
    }
  });
}

// caller needs to guarantee the lifetime of call_data.
void CompletionHandler::chat_async(ChatCallData* call_data) {
  converter_executor_.schedule([this, call_data = call_data]() {
    auto request = grpc_chat_request_to_request(call_data, tokenizer_.get());
    if (request == nullptr) {
      // TODO: finish with error
    }

    bool success = scheduler_->schedule(request);
    if (!success) {
      // TODO: finish with error
      // call_data->finish();
      // request->finish();
    }
  });
}

}  // namespace llm
