#include "completion_handler.h"

#include <glog/logging.h>
#include <grpcpp/grpcpp.h>

#include <string>
#include <thread>

#include "call_data.h"
#include "completion.grpc.pb.h"

namespace llm {

void CompletionHandler::complete_async(CompletionCallData* call_data) {
  // TODO: send the request to processor
  std::thread([call_data]() {
    const auto& request = call_data->request();
    const std::string& prompt = request.prompt();
    for (int i = 0; i < 1000; ++i) {
      CompletionResponse response;
      auto* choice = response.add_choices();
      const std::string text = prompt + std::to_string(i);
      choice->set_text(text);
      call_data->write(response);
    }
    call_data->finish();
  }).detach();
}

// caller needs to guarantee the lifetime of call_data.
void CompletionHandler::chat_async(ChatCallData* call_data) {
  // TODO: send the request to processor
  std::thread([call_data]() {
    const auto& request = call_data->request();
    for (int i = 0; i < 1000; ++i) {
      ChatResponse response;
      auto* choice = response.add_choices();
      const std::string text = std::to_string(i);
      // choice->set_text(text);
      call_data->write(response);
    }
    call_data->finish();
  }).detach();
}
}  // namespace llm
