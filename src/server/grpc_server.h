#pragma once

#include <grpcpp/grpcpp.h>

#include <string>
#include <thread>

#include "common/logging.h"
#include "handlers/chat_handler.h"
#include "handlers/completion_handler.h"
#include "handlers/models_handler.h"

namespace llm {

class GrpcServer final {
 public:
  struct Options {
    std::string address = "localhost";
    int32_t port = 8888;
  };

  GrpcServer(std::unique_ptr<CompletionHandler> completion_handler,
             std::unique_ptr<ChatHandler> chat_handler,
             std::unique_ptr<ModelsHandler> models_handler)
      : completion_handler_(std::move(completion_handler)),
        chat_handler_(std::move(chat_handler)),
        models_handler_(std::move(models_handler)) {}

  ~GrpcServer();

  bool start(const Options& options);

  void stop();

 private:
  void handle_rpcs();

  // handler for completion requests
  std::unique_ptr<CompletionHandler> completion_handler_;

  // handler for chat requests
  std::unique_ptr<ChatHandler> chat_handler_;

  // handler for models requests
  std::unique_ptr<ModelsHandler> models_handler_;

  // registed service
  Completion::AsyncService completion_service_;
  Chat::AsyncService chat_service_;

  // grpc server
  std::unique_ptr<grpc::Server> grpc_server_;
  // completion queue: the producer-consumer queue where for asynchronous server
  std::unique_ptr<grpc::ServerCompletionQueue> cq_;
  // thread for handling rpcs
  std::unique_ptr<std::thread> handler_thread_;
};

}  // namespace llm
