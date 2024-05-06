#include "utils.h"

#include <glog/logging.h>

#include <string>

#include "common.pb.h"
#include "request/request.h"

namespace llm {

RequestPriority grpc_priority_to_priority(proto::Priority priority) {
  switch (priority) {
    case proto::Priority::DEFAULT:
      return RequestPriority::MEDIUM;
    case proto::Priority::LOW:
      return RequestPriority::LOW;
    case proto::Priority::MEDIUM:
      return RequestPriority::MEDIUM;
    case proto::Priority::HIGH:
      return RequestPriority::HIGH;
    default:
      LOG(WARNING) << "Unknown priority: " << static_cast<int>(priority);
  }
  return RequestPriority::MEDIUM;
}

}  // namespace llm