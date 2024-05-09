#include "utils.h"

#include <glog/logging.h>
#include <grpcpp/grpcpp.h>

#include "common.pb.h"

namespace llm {

Priority to_priority(proto::Priority priority) {
  switch (priority) {
    case proto::Priority::DEFAULT:
      return Priority::NORMAL;
    case proto::Priority::LOW:
      return Priority::LOW;
    case proto::Priority::NORMAL:
      return Priority::NORMAL;
    case proto::Priority::HIGH:
      return Priority::HIGH;
    default:
      LOG(WARNING) << "Unknown priority: " << static_cast<int>(priority);
  }
  return Priority::NORMAL;
}

grpc::StatusCode to_grpc_status_code(StatusCode code) {
  switch (code) {
    case StatusCode::OK:
      return grpc::StatusCode::OK;
    case StatusCode::CANCELLED:
      return grpc::StatusCode::CANCELLED;
    case StatusCode::UNKNOWN:
      return grpc::StatusCode::UNKNOWN;
    case StatusCode::INVALID_ARGUMENT:
      return grpc::StatusCode::INVALID_ARGUMENT;
    case StatusCode::DEADLINE_EXCEEDED:
      return grpc::StatusCode::DEADLINE_EXCEEDED;
    case StatusCode::RESOURCE_EXHAUSTED:
      return grpc::StatusCode::RESOURCE_EXHAUSTED;
    case StatusCode::UNAUTHENTICATED:
      return grpc::StatusCode::UNAUTHENTICATED;
    case StatusCode::UNAVAILABLE:
      return grpc::StatusCode::UNAVAILABLE;
    case StatusCode::UNIMPLEMENTED:
      return grpc::StatusCode::UNIMPLEMENTED;
    default:
      LOG(WARNING) << "Unknown status code: " << static_cast<uint8_t>(code);
  }
  return grpc::StatusCode::UNKNOWN;
}

}  // namespace llm