#include <folly/init/Init.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <thread>

#include "grpc_server.h"

using namespace llm;

int main(int argc, char** argv) {
  // glog and glfag will be initialized in folly::init
  folly::Init init(&argc, &argv);

  GrpcServer server;

  GrpcServer::Options options;
  options.address = "localhost";
  options.port = 8888;

  if (!server.start(options)) {
    LOG(ERROR) << "failed to start grpc server";
    return -1;
  }

  // TODO: add graceful shutdown
  while (true) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    // TODO: update server status
  }
  server.stop();
  return 0;
}
