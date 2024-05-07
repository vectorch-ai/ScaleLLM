#pragma once

#include <glog/logging.h>

#include <optional>
#include <string>
#include <vector>

#include "status.h"

namespace llm {

// Priority of the request.
// The higher the priority, the sooner the request is processed.
enum class Priority { HIGH = 0, NORMAL, LOW };

// "stop" - the model hit a natural stop point or a provided stop sequence.
// "length" - the maximum number of tokens specified in the request was reached.
// "function_call" - the model called a function.
enum class FinishReason {
  NONE = 0,
  STOP = 1,
  LENGTH,
  FUNCTION_CALL,
};

struct Usage {
  // the number of tokens in the prompt.
  size_t num_prompt_tokens = 0;

  // the number of tokens in the generated completion.
  size_t num_generated_tokens = 0;

  // the total number of tokens used in the request (prompt + completion).
  size_t num_total_tokens = 0;
};

struct SequenceOutput {
  // the index of the sequence in the request.
  size_t index;

  // the generated/delta text.
  // delta text is the text generated since the last response for streaming.
  std::string text;

  // the reason the sequence finished.
  std::optional<std::string> finish_reason;
};

struct RequestOutput {
  RequestOutput() = default;
  
  RequestOutput(Status&& _status) : status(std::move(_status)) {}

  // the status of the request.
  std::optional<Status> status;

  // the output for each sequence in the request.
  std::vector<SequenceOutput> outputs;

  // the statistics for the request.
  std::optional<Usage> usage;

  // whether the request is finished.
  bool finished = false;
};

inline std::optional<std::string> to_string(FinishReason reason) {
  switch (reason) {
    case FinishReason::NONE:
      return std::nullopt;
    case FinishReason::STOP:
      return "stop";
    case FinishReason::LENGTH:
      return "length";
    case FinishReason::FUNCTION_CALL:
      return "function_call";
    default:
      LOG(WARNING) << "Unknown finish reason: " << static_cast<int>(reason);
  }
  return std::nullopt;
}

}  // namespace llm