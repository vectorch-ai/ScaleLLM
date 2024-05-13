#pragma once

#include <grpcpp/grpcpp.h>

#include "common.pb.h"
#include "request/output.h"
#include "request/status.h"
namespace llm {

Priority to_priority(proto::Priority priority);

grpc::StatusCode to_grpc_status_code(StatusCode code);

}  // namespace llm