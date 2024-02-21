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
constexpr size_t kMaxBatchSize = 100;

FCFSSchedulerPolicy::FCFSSchedulerPolicy(ResponseHandler* response_handler,
                                         BlockManager* block_manager)
  : response_handler_(response_handler), block_manager_(block_manager),
    waiting_queue_(kRequestQueueSize) {}

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

  running_queue_.clear();
  blocking_queue_.clear();
}

bool FCFSSchedulerPolicy::schedule(std::unique_ptr<Request>& request) {
  GCHECK(request != nullptr);
  if (waiting_queue_.write(request.get())) {
    // take over the ownership of the request
    request.release();
    return true;
  }
  // queue is full
  return false;
}

std::vector<Sequence*> FCFSSchedulerPolicy::build_batch() {
  std::vector<Request*> ready_queue;
  for (Request* request : running_queue_) {
    if (request->is_finished()) {
      response_handler_->on_request_finish(request);
    }
    ready_queue.emplace_back(request);
  }
  running_queue_.clear();

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

  std::vector<Sequence*> running_batch;
  running_batch.reserve(kMaxBatchSize);
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
      running_batch.insert(running_batch.end(),
                            sequences.begin(),
                            sequences.end());
    }
  }
  
  if (running_batch.empty() && !blocking_queue_.empty()) {
    GLOG(ERROR) << "Not enough memory to schedule one sequence";

    Request* request = blocking_queue_.back();
    blocking_queue_.pop_back();

    // TODO: optimize the logic to only release blocks for sequences one by one
    response_handler_->on_request_finish(request);
  }
  return running_batch;
}
} // llm
