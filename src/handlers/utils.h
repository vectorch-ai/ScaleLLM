#pragma once
#include <string>

#include "common.pb.h"
#include "request/request.h"

namespace llm {

Priority grpc_priority_to_priority(proto::Priority priority);

std::string finish_reason_to_string(FinishReason reason);

}  // namespace llm