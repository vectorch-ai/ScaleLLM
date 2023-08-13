#pragma once

#include <glog/logging.h>
#include <grpcpp/grpcpp.h>

#include <string>
#include <thread>

#include "completion.grpc.pb.h"

namespace llm {

class GrpcServer final {
 public:
  struct Options {
    std::string address = "localhost";
    int32_t port = 8888;
  };

  GrpcServer() = default;

  ~GrpcServer();

  bool start(const Options& options);

  void stop();

 private:
  void handle_rpcs();

  // registed service
  Completion::AsyncService service_;
  // grpc server
  std::unique_ptr<grpc::Server> grpc_server_;
  // completion queue: the producer-consumer queue where for asynchronous server
  std::unique_ptr<grpc::ServerCompletionQueue> cq_;
  // thread for handling rpcs
  std::unique_ptr<std::thread> handler_thread_;
};

}  // namespace llm
