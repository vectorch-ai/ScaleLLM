#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace llm {

// Status of the request.
enum class RequestStatus {
  // The request is waiting to be processed.
  // still waiting in the queue.
  WAITING,

  // The request is currently being processed.
  // a worker has been assigned to the request.
  PROCESSING,

  // The request has been preempted.
  // preempted usually due to a higher priority request or limit on resources.
  PREEMPTED,

  // The request has been completed.
  // usually due to reaching the maximum number of tokens or
  // reaching a stopping condition.
  COMPLETED,

  // The request has been cancelled.
  // usually due to a user request.
  CANCELLED,
};

// A request context is a data structure that encapsulates all the necessary
// information required to process a request efficiently. It acts as a
// container, holding essential data, such as input parameters, configuration
// settings, and any additional context-specific details related to the
// request's handling.
struct RequestContext {
  // The unique id of the request.
  uint64_t request_id;

  // prompt to generate completions for
  std::string prompt;

  // token ids generated from p
  std::vector<int> token_ids;

  // the length of the prompt
  int prompt_len = 0;

  // the current position in generating tokens
  int cur_pos = 0;

  // the maximum number of tokens to generate.
  int max_tokens = 0;

  // Whether to stream back partial results as they are generated.
  bool stream = false;

  // The status of the request.
  RequestStatus status = RequestStatus::WAITING;

  // Scheduled time of the request.
  uint64_t scheduled_time = 0;

  // TODO: cache related

  // TODO: sampling related
};

// Compare two request contexts by their scheduled time.
// Used for priority queue (min heap).
struct RequestContextPtrLess {
  bool operator()(const RequestContext* a, const RequestContext* b) const {
    return a->scheduled_time < b->scheduled_time;
  }
};

}  // namespace llm
