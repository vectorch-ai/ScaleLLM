#include "utils.h"

#include <glog/logging.h>

#include "common.pb.h"

namespace llm {

Priority grpc_priority_to_priority(proto::Priority priority) {
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

}  // namespace llm