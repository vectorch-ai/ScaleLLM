#pragma once

#include <glog/logging.h>
#include <grpcpp/grpcpp.h>

#include <string>
#include <thread>

#include "completion.grpc.pb.h"

namespace llm {

// Interface for the classes that are used to handle grpc requests.
class ICallData {
 public:
  virtual ~ICallData() = default;

  virtual void proceed() = 0;
};

// Class encompasing the state and logic needed to serve a request.
template <typename Request, typename Response>
class CallData : public ICallData {
 public:
  enum class Status { CREATE, PROCESS, FINISH, DESTROY };

  // callback for registering itself to the service
  using OnRegister =
      std::function<void(grpc::ServerContext* context,
                         Request* request,
                         grpc::ServerAsyncWriter<Response>* responder,
                         grpc::ServerCompletionQueue* new_call_cq,
                         grpc::ServerCompletionQueue* notification_cq,
                         void* tag)>;
  // callback for new request
  using OnRequest = std::function<void(CallData<Request, Response>*)>;

  static CallData* create(grpc::ServerCompletionQueue* cq,
                          OnRegister on_register,
                          OnRequest on_request) {
    // NOLINTNEXTLINE
    return new CallData(cq, on_register, on_request);
  }

  const Request& request() const { return request_; }

  // call following methods to reply to client
  void write(const Response& response) {
    wait_for_ops();
    status_ = Status::PROCESS;
    responder_.Write(response, this);
  }

  void write_and_finish(const Response& response,
                        const grpc::Status& status = grpc::Status::OK) {
    wait_for_ops();
    status_ = Status::FINISH;
    responder_.WriteAndFinish(response, grpc::WriteOptions(), status, this);
  }

  void finish(const grpc::Status& status = grpc::Status::OK) {
    wait_for_ops();
    status_ = Status::FINISH;
    responder_.Finish(status, this);
  }

  // proceed to the next state.
  // it is designed to be called by the grpc handler thread only
  void proceed() override {
    CHECK(ops_in_progress_.exchange(false));

    // it is notification from cq for new request
    if (status_ == Status::CREATE) {
      // Spawn a new CallData instance to serve new clients while we process
      // the one for this CallData.
      CallData::create(cq_, on_register_, on_new_request_);

      // The actual processing.
      on_new_request_(this);
      status_ = Status::PROCESS;
    } else if (status_ == Status::PROCESS) {
      // do nothing
    } else if (status_ == Status::FINISH) {
      status_ = Status::DESTROY;
      delete this;
    }
  }

 private:
  CallData(grpc::ServerCompletionQueue* cq,
           OnRegister on_register,
           OnRequest on_request)
      : cq_(cq),
        responder_(&ctx_),
        on_register_(on_register),
        on_new_request_(on_request) {
    // register itself to the service for handling request
    on_register_(&ctx_, &request_, &responder_, cq_, cq_, this);
  }

  void wait_for_ops() {
    bool expected = false;
    while (!ops_in_progress_.compare_exchange_weak(expected, true)) {
      // reset the expected value after a failed exchange
      expected = false;
      std::this_thread::yield();
    }
  }

  Status status_ = Status::CREATE;

  // completion queue: the producer-consumer queue where for asynchronous server
  // notifications.
  grpc::ServerCompletionQueue* cq_;

  // context for the request.
  grpc::ServerContext ctx_;

  // request get from client
  Request request_;

  // responder for replying to client
  grpc::ServerAsyncWriter<Response> responder_;

  // it is used to make sure write ops are not called concurrently
  std::atomic<bool> ops_in_progress_{true};

  // callback for registering itself to the service
  OnRegister on_register_;

  // callback for new request
  OnRequest on_new_request_;
};

using CompletionCallData = CallData<CompletionRequest, CompletionResponse>;
using ChatCallData = CallData<ChatRequest, ChatResponse>;

}  // namespace llm
