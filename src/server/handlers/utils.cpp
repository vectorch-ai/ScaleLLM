#include "utils.h"

#include <string>

#include "common.pb.h"
#include "common/logging.h"
#include "request/request.h"

namespace llm {

RequestPriority grpc_priority_to_priority(Priority priority) {
  switch (priority) {
    case Priority::DEFAULT:
      return RequestPriority::MEDIUM;
    case Priority::LOW:
      return RequestPriority::LOW;
    case Priority::MEDIUM:
      return RequestPriority::MEDIUM;
    case Priority::HIGH:
      return RequestPriority::HIGH;
    default:
      GLOG(WARNING) << "Unknown priority: " << static_cast<int>(priority);
  }
  return RequestPriority::MEDIUM;
}

std::string finish_reason_to_string(FinishReason reason) {
  switch (reason) {
    case FinishReason::NONE:
      return "";
    case FinishReason::STOP:
      return "stop";
    case FinishReason::LENGTH:
      return "length";
    case FinishReason::FUNCTION_CALL:
      return "function_call";
    default:
      GLOG(WARNING) << "Unknown finish reason: " << static_cast<int>(reason);
  }
  return "";
}

}  // namespace llm