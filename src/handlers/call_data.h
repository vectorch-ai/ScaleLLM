#pragma once

#include <folly/MPMCQueue.h>
#include <grpcpp/alarm.h>
#include <grpcpp/grpcpp.h>

#include <atomic>
#include <memory>
#include <optional>
#include <string>

namespace llm {

constexpr size_t kResponseQueueSize = 20;

// Interface for the classes that are used to handle grpc requests.
class ICallData {
 public:
  virtual ~ICallData() = default;

  // returns true if the rpc is ok and the call data is not finished
  // returns false if the call data is finished and can be deleted
  virtual bool proceed(bool rpc_ok) = 0;
};

// Class encompasing the state and logic needed to serve a server streaming
// request.
template <typename Request, typename Response>
class CallData : public ICallData {
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
  using OnRequest = std::function<void(CallData<Request, Response>*)>;

  CallData(grpc::ServerCompletionQueue* cq,
           OnRegister on_register,
           OnRequest on_request)
      : cq_(cq),
        responder_(&ctx_),
        on_register_(on_register),
        on_new_request_(on_request),
        response_queue_(kResponseQueueSize) {
    // register itself to the service for handling request
    on_register_(&ctx_, &request_, &responder_, cq_, cq_, this);
  }

  const Request& request() const { return request_; }

  // call following methods to reply to client
  // returns true if the response has been accepted and will be delivered
  // asynchronously.
  // returns false if the rpc channel has been closed/cancelled.
  bool write(Response response) {
    if (!rpc_ok_.load(std::memory_order_relaxed)) {
      return false;
    }

    // pack the response with state
    auto response_with_state =
        std::make_unique<ResponseWithState>(std::move(response));

    // response_queue take the ownership of the response
    response_queue_.blockingWrite(response_with_state.release());
    // notify the grpc handler thread
    write_alarm_.Set(cq_, gpr_now(gpr_clock_type::GPR_CLOCK_REALTIME), this);
    return true;
  }

  // returns false if the rpc channel has been closed/cancelled.
  bool finish_with_error(const grpc::StatusCode& code,
                         const std::string& error_message) {
    return finish(grpc::Status(code, error_message));
  }

  // returns false if the rpc channel has been closed/cancelled.
  bool finish(const grpc::Status& grpc_status = grpc::Status::OK) {
    // pack status with grpc status
    auto response_with_state = std::make_unique<ResponseWithState>(grpc_status);
    // response_queue take the ownership of the response
    response_queue_.blockingWrite(response_with_state.release());
    // notify the grpc handler thread
    finish_alarm_.Set(cq_, gpr_now(gpr_clock_type::GPR_CLOCK_REALTIME), this);
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
      // rpc error before acctually processing the request, release the calldata
      if (!rpc_ok) {
        return false;
      }

      // Spawn a new CallData instance to serve new clients while we process
      // the one for this CallData.
      new CallData(cq_, on_register_, on_new_request_);

      // set status to WRITE to process response
      status_ = Status::WRITE;
      // The actual processing.
      on_new_request_(this);
    } else if (status_ == Status::WRITE) {
      // pull the next request from the queue and send it to client
      ResponseWithState* r = nullptr;
      if (response_queue_.read(r)) {
        std::unique_ptr<ResponseWithState> rs(r);
        if (rs->response.has_value()) {
          // if rpc is ok, send the response to client, otherwise, wait for
          // finish alarm
          if (rpc_ok) {
            // change the status to pending to wait for write op to finish
            status_ = Status::PENDING;
            responder_.Write(rs->response.value(), this);
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

  // write alarm for notifying the grpc handler thread
  grpc::Alarm write_alarm_;

  // finish alarm for notifying the grpc handler thread
  grpc::Alarm finish_alarm_;

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

  // a thread safe queue of response, bounded by kResponseQueueSize
  // the call data owns the responses and manages their lifetimes.
  folly::MPMCQueue<ResponseWithState*> response_queue_;
};

}  // namespace llm
