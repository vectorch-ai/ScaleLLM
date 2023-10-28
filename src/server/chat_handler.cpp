#include "chat_handler.h"

#include <glog/logging.h>
#include <grpcpp/grpcpp.h>
#include <torch/torch.h>

#include <string>
#include <thread>

#include "call_data.h"
#include "chat.grpc.pb.h"
#include "request/request.h"

namespace llm {

namespace {

std::unique_ptr<Request> grpc_request_to_request(ChatCallData* call_data,
                                                 const Tokenizer* tokenizer) {
  auto request = std::make_unique<Request>();
  return request;
}

}  // namespace

ChatHandler::ChatHandler(Scheduler* scheduler, const Engine* engine)
    : scheduler_(scheduler) {
  CHECK(scheduler_ != nullptr);
  tokenizer_ = engine->tokenizer();
  model_args_ = engine->model_args();
}

ChatHandler::~ChatHandler() {}

void ChatHandler::chat_async(ChatCallData* call_data) {
  converter_executor_.schedule([this, call_data = call_data]() {
    auto request = grpc_request_to_request(call_data, tokenizer_.get());
    if (request == nullptr) {
      // TODO: finish with error
      call_data->finish(grpc::Status::OK);
      return;
    }

    bool success = scheduler_->schedule(request);
    if (!success) {
      // TODO: finish with error: out of capacity
      call_data->finish(grpc::Status::OK);
    }
  });
}

}  // namespace llm
