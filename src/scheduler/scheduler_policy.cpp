#include "scheduler_policy.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>

#include "common/logging.h"
#include "memory/block_manager.h"
#include "request/request.h"
#include "request/sequence.h"
#include "scheduler/response_handler.h"

namespace llm {
constexpr size_t kRequestQueueSize = 100000;
// TODO: reader from config
constexpr size_t kMaxBatchSize = 100;

constexpr uint64_t kStepSleepTimeMs = 10;

DEFINE_int32(streaming_token_buffer_size,
             1,
             "number of tokens to buffer before streaming to client");

FCFSSchedulerPolicy::FCFSSchedulerPolicy(ResponseHandler* response_handler,
                                         BlockManager* block_manager)
  : response_handler_(response_handler), block_manager_(block_manager) {}

FCFSSchedulerPolicy::~FCFSSchedulerPolicy() {
  // release all requests in the queue
  while (!waiting_queue_.isEmpty()) {
    Request* request = nullptr;
    waiting_queue_.read(request);
    std::unique_ptr<Request> request_ptr(request);
  }

  // release all requests in the ready queue
  for (Request* request : blocking_queue_) {
    std::unique_ptr<Request> request_ptr(request);
  }

  // release all requests in the running queue
  for (Request* request : running_queue_) {
    std::unique_ptr<Request> request_ptr(request);
  }

  running_batch_.clear();
  running_queue_.clear();
  blocking_queue_.clear();
}

bool FCFSSchedulerPolicy::try_emplace(std::unique_ptr<Request>& request) {
  GCHECK(request != nullptr);
  if (waiting_queue_.write(request.get())) {
    // take over the ownership of the request
    request.release();
    return true;
  }
  // queue is full
  return false;
}

void FCFSSchedulerPolicy::schedule() {
  std::vector<Request*> ready_queue;
  for (Request* request : running_queue_) {
    if (request->is_finished()) {
      response_handler_->on_request_finish(request);
    }

    ready_queue.emplace_back(request);
  }

  running_queue_.clear();
  running_batch_.clear();

  for (Request* request : blocking_queue_) {
    ready_queue.emplace_back(request);
  }
  blocking_queue_.clear();

  while (!waiting_queue_.isEmpty()) {
    Request* request = nullptr;
    waiting_queue_.read(request);
    GCHECK(request != nullptr);
    ready_queue.emplace_back(request);
  }

  for (Request* request : ready_queue) {
    if (request->sequences.empty() || request->is_finished()) {
      continue;
    }

    std::vector<Sequence*> sequences;
    sequences.reserve(request->sequences.size());
    for (Sequence& sequence : request->sequences) {
      if (sequence.is_finished()) {
        continue;
      }
      if (block_manager_->allocate_slots_for_sequence(&sequence)) {
        sequences.emplace_back(&sequence);
      }
    }
    if (sequences.empty()) {
      blocking_queue_.emplace_back(request);
    } else {
      running_queue_.emplace_back(request);
      running_batch_.insert(running_batch_.end(),
                             sequences.begin(),
                             sequences.end());
    }
  }
  
  if (running_batch_.empty() && !blocking_queue_.empty()) {
    // don't have enough memory to schedule one sequence
    GLOG(ERROR) << "Not enough memory to schedule one sequence";

    Request* request = blocking_queue_.back();
    blocking_queue_.pop_back();

    // TODO: optimize the logic to only release blocks for sequences one by one
    response_handler_->on_request_finish(request);
  }
}
} // llm
