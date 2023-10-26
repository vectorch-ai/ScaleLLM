#include <gflags/gflags.h>
#include <glog/logging.h>
#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include <grpcpp/support/sync_stream.h>

#include <iostream>
#include <memory>
#include <string>

#include "completion.grpc.pb.h"
#include "completion.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::ClientReader;
using grpc::Status;

using llm::Completion;
using llm::CompletionRequest;
using llm::CompletionResponse;

DEFINE_string(priority, "DEFAULT", "priority of the request, DEFAULT, LOW, MEDIUM, HIGH");

DEFINE_double(temperature, 0.6, "Temperature for sampling.");

DEFINE_double(top_p, 0.9, "Top p for sampling.");
DEFINE_int32(max_tokens, 256, "Maximum number of tokens to generate.");

// DEFINE_double(repetition_penalty, 1.0, "Repetition penalty for sampling.");

DEFINE_double(frequency_penalty, 0.0, "Frequency penalty for sampling.");
DEFINE_double(presence_penalty, 0.0, "Presence penalty for sampling.");

class ChatClient final {
 public:
  ChatClient(std::shared_ptr<Channel> channel)
      : stub_(Completion::NewStub(channel)) {}

  void send_and_receive(const std::string& prompt) {
    // Create a message to send to the server
    CompletionRequest request;
    request.set_prompt(prompt);
    request.set_temperature(FLAGS_temperature);
    request.set_top_p(FLAGS_top_p);
    request.set_frequency_penalty(FLAGS_frequency_penalty);
    request.set_presence_penalty(FLAGS_presence_penalty);
    request.set_max_tokens(FLAGS_max_tokens);

    llm::Priority priority{};
    CHECK(llm::Priority_Parse(FLAGS_priority, &priority));
    request.set_priority(priority);

    // Create a stream for receiving messages
    ClientContext context;
    std::unique_ptr<grpc::ClientReader<CompletionResponse>> reader(
        stub_->Complete(&context, request));

    CompletionResponse message;
    while (reader->Read(&message)) {
      // LOG(ERROR) << "got response: " << message.DebugString();
      // pretty print the response
      for (const auto& choice : message.choices()) {
        // LOG(INFO) << choice.DebugString();
        std::cout << choice.text() << std::flush;
      }
    }

    Status status = reader->Finish();
    if (!status.ok()) {
      LOG(ERROR) << "RPC failed, error code: " << status.error_code()
                 << ", error message: " << status.error_message()
                 << ", error details: " << status.error_details();
    }
  }

 private:
  std::unique_ptr<Completion::Stub> stub_;
};

int main(int argc, char* argv[]) {
  // initialize glog and gflags
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Define the server address and port
  std::string server_address("localhost:8888");

  // Create a gRPC channel
  auto channel =
      grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());

  // Create a chat client
  ChatClient client(channel);

  std::string prompt = "Enter a prompt: ";
  std::cout << prompt;

  std::string input;
  while (std::getline(std::cin, input) && input != "exit") {
    if (input.empty()) {
      continue;
    }
    client.send_and_receive(input);
    std::cout << std::endl << prompt;
  }
  return 0;
}
