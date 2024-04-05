#pragma once

#include <glog/logging.h>
#include <grpcpp/alarm.h>
#include <grpcpp/grpcpp.h>

#include <atomic>
#include <memory>
#include <optional>
#include <string>

namespace llm {

// Interface for the classes that are used to handle grpc requests.
class CallData {
 public:
  virtual ~CallData() = default;

  // returns true if the rpc is ok and the call data is not finished
  // returns false if the call data is finished and can be deleted
  virtual bool proceed(bool rpc_ok) = 0;
};

// Class encompasing the state and logic needed to serve a server streaming
// request.
template <typename Request, typename Response>
class StreamCallData : public CallData {
 public:
  enum class Status { CREATE, WRITE, PENDING, FINISH };

  // pack the response with state
  struct ResponseWithState {
    ResponseWithState(Response _response) : response(std::move(_response)) {}

    ResponseWithState(grpc::Status _grpc_status) : grpc_status(_grpc_status) {}

    // response to be sent to client
    std::optional<Response> response;
    // grpc status to be sent to client
    grpc::Status grpc_status = grpc::Status::OK;
  };

  // callback for registering itself to the service
  using OnRegister =
      std::function<void(grpc::ServerContext* context,
                         Request* request,
                         grpc::ServerAsyncWriter<Response>* responder,
                         grpc::ServerCompletionQueue* new_call_cq,
                         grpc::ServerCompletionQueue* notification_cq,
                         void* tag)>;
  // callback for new request
  using OnRequest = std::function<void(StreamCallData<Request, Response>*)>;

  StreamCallData(grpc::ServerCompletionQueue* cq,
                 OnRegister on_register,
                 OnRequest on_request)
      : cq_(cq),
        responder_(&ctx_),
        on_register_(on_register),
        on_new_request_(on_request) {
    // register itself to the service for handling request
    on_register_(&ctx_, &request_, &responder_, cq_, cq_, this);
  }

  const Request& request() const { return request_; }

  // returns true if the rpc is ok
  bool is_rpc_ok() const { return rpc_ok_.load(std::memory_order_relaxed); }

  // call following methods to reply to client
  // returns true if the response has been accepted and will be delivered
  // asynchronously.
  // returns false if the rpc channel has been closed/cancelled.
  bool write(Response response) {
    // pack the response with state
    auto new_response =
        std::make_shared<ResponseWithState>(std::move(response));
    // wait previous response to be processed
    std::shared_ptr<ResponseWithState> expected = nullptr;
    while (!std::atomic_compare_exchange_weak(
        &response_, &expected, new_response)) {
      expected = nullptr;
    }

    // notify the grpc handler thread
    notify_alarm_.Set(
        cq_, gpr_time_0(gpr_clock_type::GPR_CLOCK_MONOTONIC), this);
    return rpc_ok_.load(std::memory_order_relaxed);
  }

  // returns false if the rpc channel has been closed/cancelled.
  bool finish_with_error(const grpc::StatusCode& code,
                         const std::string& error_message) {
    return finish(grpc::Status(code, error_message));
  }

  // returns false if the rpc channel has been closed/cancelled.
  bool finish(const grpc::Status& grpc_status = grpc::Status::OK) {
    // pack status with grpc status
    auto new_response = std::make_shared<ResponseWithState>(grpc_status);
    // wait previous response to be processed
    std::shared_ptr<ResponseWithState> expected = nullptr;
    while (!std::atomic_compare_exchange_weak(
        &response_, &expected, new_response)) {
      expected = nullptr;
    }

    // notify the grpc handler thread
    notify_alarm_.Set(
        cq_, gpr_time_0(gpr_clock_type::GPR_CLOCK_MONOTONIC), this);
    return rpc_ok_.load(std::memory_order_relaxed);
  }

  // proceed to the next state.
  bool proceed(bool rpc_ok) override {
    // record the rpc status
    if (!rpc_ok) {
      rpc_ok_.store(false, std::memory_order_relaxed);
    }
    // rpc is ok only if all the ops are ok
    rpc_ok = rpc_ok && rpc_ok_.load(std::memory_order_relaxed);

    // it is notification from cq for new request
    if (status_ == Status::CREATE) {
      // Spawn a new CallData instance to serve new clients
      new StreamCallData(cq_, on_register_, on_new_request_);

      // rpc error before acctually processing the request, release the calldata
      if (!rpc_ok) {
        return false;
      }

      // set status to WRITE to process response
      status_ = Status::WRITE;
      // The actual processing.
      on_new_request_(this);
    } else if (status_ == Status::WRITE) {
      if (auto rs = std::atomic_load(&response_)) {
        // send the response to client
        if (rs->response.has_value()) {
          // if rpc is ok, send the response to client, otherwise, wait for
          // finish alarm
          if (rpc_ok) {
            // change the status to pending to wait for write op to finish
            status_ = Status::PENDING;
            responder_.Write(rs->response.value(), this);
          } else {
            std::atomic_store(&response_, {});
          }
        } else {
          if (rpc_ok) {
            // the rpc is ok, send the finish status to client
            status_ = Status::FINISH;
            responder_.Finish(rs->grpc_status, this);
          } else {
            // the request has been finished, release the calldata
            return false;
          }
        }
      }
    } else if (status_ == Status::PENDING) {
      // the write op has been finished, proceed to the next write op
      status_ = Status::WRITE;
      std::atomic_store(&response_, {});
    } else if (status_ == Status::FINISH) {
      // Once in the FINISH state, deallocate CallData.
      return false;
    }
    return true;
  }

 private:
  Status status_ = Status::CREATE;

  // completion queue: the producer-consumer queue where for asynchronous server
  // notifications.
  grpc::ServerCompletionQueue* cq_;

  // context for the request.
  grpc::ServerContext ctx_;

  // alarm for notifying the grpc handler thread
  grpc::Alarm notify_alarm_;

  // request get from client
  Request request_;

  // responder for replying to client
  grpc::ServerAsyncWriter<Response> responder_;

  // it is used to record the rpc status
  std::atomic<bool> rpc_ok_{true};

  // callback for registering itself to the service
  OnRegister on_register_;

  // callback for new request
  OnRequest on_new_request_;

  // next response to be sent to client
  std::shared_ptr<ResponseWithState> response_;
};

}  // namespace llm
