#include "grpc_server.h"

#include <absl/strings/str_format.h>
#include <grpcpp/grpcpp.h>

#include <memory>

namespace llm {

// This class can be moved to cpp file
// Class encompasing the state and logic needed to serve a request.
class CallData {
 public:
  static CallData* create(Completion::AsyncService* service,
                          grpc::ServerCompletionQueue* cq) {
    // NOLINTNEXTLINE
    return new CallData(service, cq);
  }

  // proceed to the next state
  void proceed() {
    if (status_ == PROCESS) {
      if (times_called_++ == 0) {
        // Spawn a new CallData instance to serve new clients while we process
        // the one for this CallData.
        CallData::create(service_, cq_);
      }

      // The actual processing.
      // TODO: send the request to processor
      // OnNewRequest(this);
    } else {
      CHECK_EQ(status_, FINISH);
      delete this;
    }
  }

  // call following methods to reply to client
  void write(const CompletionResponse& response) {
    // TODO: check if previous write is finished
    // for each task from the completion queue, Write can be called only once.
    responder_.Write(response, this);
  }

  void write_and_finish(const CompletionResponse& response) {
    status_ = FINISH;
    responder_.WriteAndFinish(
        response, grpc::WriteOptions(), grpc::Status::OK, this);
  }

  void finish() {
    status_ = FINISH;
    responder_.Finish(grpc::Status::OK, this);
  }

 private:
  CallData(Completion::AsyncService* service, grpc::ServerCompletionQueue* cq)
      : service_(service), cq_(cq), responder_(&ctx_) {
    // register itself to the service for handling request
    service_->RequestComplete(&ctx_, &request_, &responder_, cq_, cq_, this);
  }

  // The number of times the proceed method has been called.
  size_t times_called_ = 0;

  // async service
  Completion::AsyncService* service_;

  // completion queue: the producer-consumer queue where for asynchronous server
  // notifications.
  grpc::ServerCompletionQueue* cq_;

  // context for the request.
  grpc::ServerContext ctx_;

  // request get from client
  CompletionRequest request_;

  // responder for replying to client
  grpc::ServerAsyncWriter<CompletionResponse> responder_;

  // status of the request
  enum CallStatus { PROCESS, FINISH };
  // The current serving state.
  CallStatus status_{};
};

GrpcServer::~GrpcServer() { stop(); }

bool GrpcServer::start(const Options& options) {
  std::string server_address =
      absl::StrFormat("%s:%d", options.address, options.port);

  grpc::ServerBuilder builder;
  // TODO: add authentication credentials
  // Listen on the given address without any authentication mechanism.
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  // Register "service_" as the instance through which we'll communicate with
  // clients. In this case it corresponds to an *asynchronous* service.
  builder.RegisterService(&service_);
  // Get hold of the completion queue used for the asynchronous communication
  // with the gRPC runtime.
  cq_ = builder.AddCompletionQueue();
  // Finally assemble the server.
  grpc_server_ = builder.BuildAndStart();
  LOG(INFO) << "Server listening on " << server_address;

  // Spawn a new CallData instance to serve new clients.
  // TODO: spawn multiple CallData instances and support multi-threading
  CallData::create(&service_, cq_.get());

  // Proceed to the server's main loop.
  handler_thread_ = std::make_unique<std::thread>([this]() { handle_rpcs(); });
  return true;
}

void GrpcServer::stop() {
  if (grpc_server_) {
    grpc_server_->Shutdown();
    grpc_server_.reset();
  }
  // Always shutdown the completion queue after the server.
  if (cq_) {
    cq_->Shutdown();
    cq_.reset();
  }

  if (handler_thread_ && handler_thread_->joinable()) {
    handler_thread_->join();
  }
}

// This can be run in multiple threads if needed.
void GrpcServer::handle_rpcs() {
  void* tag = nullptr;  // uniquely identifies a request.
  bool ok = false;

  // Block waiting to read the next event from the completion queue.
  // returns false if there is any kind of event or cq_ is shutting down.
  while (cq_->Next(&tag, &ok)) {
    GPR_ASSERT(ok);
    static_cast<CallData*>(tag)->proceed();
  }
}

}  // namespace llm
