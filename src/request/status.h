#pragma once

#include <cstdint>
#include <string>

namespace llm {

enum class StatusCode : uint8_t {
  // Not an error; returned on success.
  OK = 0,
  // The request was cancelled. (by user/server)
  CANCELLED = 1,
  // Unknown error.
  UNKNOWN = 2,
  // Client specified an invalid argument.
  INVALID_ARGUMENT = 3,
  // Deadline expired before operation could complete. for example, timeout.
  DEADLINE_EXCEEDED = 4,
  // Some resource has been exhausted.
  RESOURCE_EXHAUSTED = 5,
  // The request does not have valid authentication credentials.
  UNAUTHENTICATED = 6,
  // The service is currently unavailable.
  UNAVAILABLE = 7,
  // Not implemented or not supported in this service.
  UNIMPLEMENTED = 8,
};

class Status final {
 public:
  Status() = default;

  Status(StatusCode code, std::string msg)
      : code_(code), msg_(std::move(msg)) {}

  StatusCode error_code() const { return code_; }
  const std::string& error_msg() const { return msg_; }

  bool ok() const { return code_ == StatusCode::OK;}

 private:
  StatusCode code_ = StatusCode::OK;
  std::string msg_;
};

// inline std::ostream& operator<<(std::ostream& os, const Status& status) {
//   os << "Status, code: " << status.error_code() << ", message: " << status.error_msg();
//   return os;
// }


}  // namespace llm
