#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "sequence.h"

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

// Priority of the request.
// The higher the priority, the sooner the request is processed.
enum class RequestPriority { HIGH = 0, MEDIUM, LOW };

struct SamplingParameter {
  float frequency_penalty = 0.0;
  float presence_penalty = 0.0;
  float repetition_penalty = 1.0;
  float temperature = 1.0;
  float top_p = 1.0;
  int64_t top_k = 0;
  bool do_sample = false;
  uint64_t seed = 0;
};

// A request is a data structure that encapsulates all the necessary
// information required to process a request efficiently. It acts as a
// container, holding essential data, such as input parameters, configuration
// settings, and any additional context-specific details related to the
// request's handling.
struct Request {
  // The unique id of the request.
  uint64_t request_id = 0;

  // list of sequences to generate completions for the prompt
  std::vector<Sequence> sequences;

  // sampling parameters
  SamplingParameter sampling_param;

  // the maximum number of tokens to generate.
  int max_tokens = 0;

  // Whether to stream back partial results as they are generated.
  bool stream = false;

  // The status of the request.
  RequestStatus status = RequestStatus::WAITING;

  // the priority of the request.
  RequestPriority priority = RequestPriority::MEDIUM;

  // Scheduled time of the request.
  uint64_t scheduled_time = 0;

  // this function will be called when the request is finished.
  void finish();

  bool is_finished() const { return status == RequestStatus::COMPLETED; }
};

// Compare two request contexts based on priority then scheduled time.
// if a < b then a should be processed before b.
struct RequestPtrLess {
  bool operator()(const Request* a, const Request* b) const {
    if (a->priority == b->priority) {
      return a->scheduled_time < b->scheduled_time;
    }
    return a->priority < b->priority;
  }
};

// Compare two request contexts based on priority then scheduled time.
// if a > b then a should be processed after b.
struct RequestPtrGreater {
  bool operator()(const Request* a, const Request* b) const {
    if (a->priority == b->priority) {
      return a->scheduled_time > b->scheduled_time;
    }
    return a->priority > b->priority;
  }
};

}  // namespace llm
