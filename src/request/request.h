#pragma once

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include "sampling/parameters.h"
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

struct Statistics {
  // the number of tokens in the prompt.
  size_t num_prompt_tokens = 0;
  // the number of tokens in the generated completion.
  size_t num_generated_tokens = 0;
  // the total number of tokens used in the request (prompt + completion).
  size_t num_total_tokens = 0;
};

// Priority of the request.
// The higher the priority, the sooner the request is processed.
enum class RequestPriority { HIGH = 0, MEDIUM, LOW };

struct SequenceOutput {
  std::string text;

  FinishReason finish_reason;
};

// Function to call when a request is finished.
using OnFinish =
    std::function<bool(const std::vector<SequenceOutput>& seq_results,
                       const Status& status,
                       const Statistics& stats)>;

using OnStreamDelta =
    std::function<bool(size_t index, const SequenceDeltaOutput& output)>;

// Function to call when a stream request is finished.
using OnStreamFinish = std::function<bool(const Status& status)>;

// Function to check rpc health.
using IsRpcOK = std::function<bool()>;

// A request is a data structure that encapsulates all the necessary
// information required to process a request efficiently. It acts as a
// container, holding essential data, such as input parameters, configuration
// settings, and any additional context-specific details related to the
// request's handling.
struct Request final {
 public:
  // caller needs to gurantee prompt's lifecycle
  Request(const std::string& id,
          const std::string_view& prompt,
          size_t n,
          const std::vector<int32_t>& prompt_tokens);

  Request(const std::string& id, const std::vector<int32_t>& prompt_tokens);

  void add_sequence();

  bool is_finished() const;

  bool is_cancelled() const;

  size_t num_prompt_tokens() const { return prompt_tokens.size(); }

  bool should_expand_sequences() const;

  void expand_sequences();

  // The unique id of the request.
  // NOLINTNEXTLINE
  const std::string id;

  // Scheduled time of the request.
  // NOLINTNEXTLINE
  const int64_t created_time;

  // prompt text string
  // NOLINTNEXTLINE
  const std::string_view prompt;

  // the number of sequences to generate completions for the prompt.
  // NOLINTNEXTLINE
  const size_t num_seqs;

  // the token ids from request's prompt.
  // NOLINTNEXTLINE
  const std::vector<int32_t> prompt_tokens;

  // sampling parameters
  SamplingParameter sampling_param;

  // stopping criteria
  StoppingCriteria stopping_criteria;

  // Whether to stream back partial results as they are generated.
  bool stream = false;

  // Whether to echo back the prompt in the output.
  bool echo = true;

  // the priority of the request.
  RequestPriority priority = RequestPriority::MEDIUM;

  // list of sequences to generate completions for the prompt
  // use deque instead of vector to avoid no-copy move for Sequence
  std::deque<Sequence> sequences;

  // function to call when the request is finished.
  OnFinish on_finish;

  // function to call when a delta is generated.
  OnStreamDelta on_stream_delta;

  // function to call when a stream request is finished.
  OnStreamFinish on_stream_finish;

  // function to check rpc health.
  IsRpcOK is_rpc_ok;
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
