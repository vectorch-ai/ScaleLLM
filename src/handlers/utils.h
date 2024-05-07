#pragma once

#include "common.pb.h"
#include "request/output.h"

namespace llm {

Priority grpc_priority_to_priority(proto::Priority priority);

}  // namespace llm