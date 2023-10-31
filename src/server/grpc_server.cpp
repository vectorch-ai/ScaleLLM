#include "grpc_server.h"

#include <absl/strings/str_format.h>
#include <grpcpp/grpcpp.h>

#include <memory>
#include <thread>

#include "call_data.h"
#include "common/logging.h"

namespace llm {

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
  builder.RegisterService(&completion_service_);
  builder.RegisterService(&chat_service_);
  builder.RegisterService(models_handler_.get());
  // Get hold of the completion queue used for the asynchronous communication
  // with the gRPC runtime.
  cq_ = builder.AddCompletionQueue();
  // Finally assemble the server.
  grpc_server_ = builder.BuildAndStart();
  GLOG(INFO) << "Server listening on " << server_address;

  // Spawn a new CallData instance for complete request
  {
    auto on_register =
        [this](grpc::ServerContext* context,
               CompletionRequest* request,
               grpc::ServerAsyncWriter<CompletionResponse>* responder,
               grpc::ServerCompletionQueue* new_call_cq,
               grpc::ServerCompletionQueue* notification_cq,
               void* tag) {
          completion_service_.RequestComplete(
              context, request, responder, new_call_cq, notification_cq, tag);
        };
    auto on_request = [this](CompletionCallData* call_data) {
      completion_handler_->complete_async(call_data);
    };
    // Spawn new CallData instances to serve new clients
    new CompletionCallData(cq_.get(), on_register, on_request);
  }

  // Spawn a new CallData instance for chat request
  {
    auto on_register = [this](grpc::ServerContext* context,
                              ChatRequest* request,
                              grpc::ServerAsyncWriter<ChatResponse>* responder,
                              grpc::ServerCompletionQueue* new_call_cq,
                              grpc::ServerCompletionQueue* notification_cq,
                              void* tag) {
      chat_service_.RequestComplete(
          context, request, responder, new_call_cq, notification_cq, tag);
    };
    auto on_request = [this](ChatCallData* call_data) {
      chat_handler_->chat_async(call_data);
    };
    new ChatCallData(cq_.get(), on_register, on_request);
  }

  // Proceed to the server's main loop.
  handler_thread_ = std::make_unique<std::thread>([this]() { handle_rpcs(); });
  return true;
}

void GrpcServer::stop() {
  if (grpc_server_) {
    grpc_server_->Shutdown();
  }
  // Always shutdown the completion queue after the server.
  if (cq_) {
    cq_->Shutdown();
  }

  // wait for the handler thread to drain event queue
  if (handler_thread_ && handler_thread_->joinable()) {
    handler_thread_->join();
  }

  // release resources
  grpc_server_.reset();
  cq_.reset();
  handler_thread_.reset();
}

// This can be run in multiple threads if needed.
void GrpcServer::handle_rpcs() {
  void* tag = nullptr;  // uniquely identifies a request.
  bool rpc_ok = false;

  // Block waiting to read the next event from the completion queue.
  // returns if there is any kind of event or cq_ is shutting down.
  while (cq_->Next(&tag, &rpc_ok)) {
    ICallData* call_data = static_cast<ICallData*>(tag);
    if (!call_data->proceed(rpc_ok)) {
      // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
      delete call_data;
    }
  }
}

}  // namespace llm
