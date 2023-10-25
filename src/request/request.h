#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "sampling_parameter.h"
#include "sequence.h"
#include "status.h"
#include "stopping_criteria.h"

namespace llm {

// Status of the request.
enum class ScheduleStatus {
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

using OnFinish =
    std::function<void(const std::string& output_text, const Status& status)>;

// A request is a data structure that encapsulates all the necessary
// information required to process a request efficiently. It acts as a
// container, holding essential data, such as input parameters, configuration
// settings, and any additional context-specific details related to the
// request's handling.
struct Request {
 public:
  Request() = default;

  void add_sequence(std::string prompt,
                    std::vector<int32_t> token_ids,
                    OnStream on_stream) {
    sequences.emplace_back(std::move(prompt),
                           std::move(token_ids),
                           &sampling_param,
                           &stopping_criteria,
                           on_stream,
                           echo);
  }
  // The unique id of the request.
  std::string id;

  // list of sequences to generate completions for the prompt
  std::vector<Sequence> sequences;

  // sampling parameters
  SamplingParameter sampling_param;

  // stopping criteria
  StoppingCriteria stopping_criteria;

  // Whether to stream back partial results as they are generated.
  bool stream = false;

  // Whether to echo back the prompt in the output.
  bool echo = true;

  // The status of the request.
  // ScheduleStatus status = ScheduleStatus::WAITING;

  // the priority of the request.
  RequestPriority priority = RequestPriority::MEDIUM;

  // Scheduled time of the request.
  int64_t created_time = 0;

  // function to call when the request is finished.
  OnFinish on_finish;

  bool is_finished() const {
    for (const auto& seq : sequences) {
      if (!seq.is_finished()) {
        return false;
      }
    }
    return true;
  }
};

// Compare two request contexts based on priority then scheduled time.
// if a < b then a should be processed before b.
struct RequestPtrLess {
  bool operator()(const Request* a, const Request* b) const {
    if (a->priority == b->priority) {
      return a->created_time < b->created_time;
    }
    return a->priority < b->priority;
  }
};

// Compare two request contexts based on priority then scheduled time.
// if a > b then a should be processed after b.
struct RequestPtrGreater {
  bool operator()(const Request* a, const Request* b) const {
    if (a->priority == b->priority) {
      return a->created_time > b->created_time;
    }
    return a->priority > b->priority;
  }
};

}  // namespace llm
