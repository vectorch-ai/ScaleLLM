#include "continuous_batching_scheduler.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <folly/MPMCQueue.h>

#include <cstdint>
#include <memory>

#include "request/request.h"

namespace llm {
constexpr size_t kRequestQueueSize = 100000;
// TODO: reader from config
constexpr size_t kMaxBatchSize = 100;

ContinuousBatchingScheduler::ContinuousBatchingScheduler()
    : request_queue_(kRequestQueueSize) {}

bool ContinuousBatchingScheduler::schedule(std::unique_ptr<Request>& request) {
  CHECK(request != nullptr);
  if (request_queue_.write(request.get())) {
    // take over the ownership of the request
    request.release();
    return true;
  }
  // queue is full
  return false;
}

std::vector<Request*> ContinuousBatchingScheduler::get_batch() {
  // propogate new requests to priority_queue_
  while (!request_queue_.isEmpty()) {
    Request* request = nullptr;
    // read from request then then push to priority queue
    request_queue_.read(request);
    CHECK(request != nullptr);
    priority_queue_.push(request);
  }

  // add requests in current batch back to the priority queue
  std::unordered_map<Request*, size_t> request_to_idx;
  for (size_t i = 0; i < batch_.size(); ++i) {
    Request* request = batch_[i];
    request_to_idx[request] = i;
    priority_queue_.push(request);
  }

  std::vector<Request*> new_batch;
  // request in [begin_idx, end_idx) are in current batch but not in new batch.
  size_t begin_idx = 0;
  size_t end_idx = batch_.size();
  while (!priority_queue_.empty()) {
    Request* candidate = priority_queue_.top();
    // no more slots available
    if (!block_manager_->allocate_slots_for_request(candidate)) {
      // try to preempt requests in current batch
      if (begin_idx == end_idx) {
        // no requests left to preempt
        break;
      }
      // preempt the lowest priority (last) request in current batch
      CHECK(end_idx > begin_idx);
      Request* preempted_request = batch_[--end_idx];
      block_manager_->release_slots_for_request(preempted_request);
      continue;
    }
    // update index range for current batch
    auto it = request_to_idx.find(candidate);
    if (it != request_to_idx.end()) {
      CHECK(it->second >= begin_idx);
      begin_idx = it->second;
    }

    // add candidate to new batch
    new_batch.push_back(candidate);
    priority_queue_.pop();
  }
  CHECK(begin_idx == end_idx);
  return new_batch;
}

// step the scheduler forward by one step
// may get blocked if there are no requests to process
void ContinuousBatchingScheduler::step(const absl::Duration& timeout) {
  // get a new batch of requests
  batch_ = get_batch();
  engine_->forward(batch_);

  // TODO: process finished requests
  for (auto& request : batch_) {
    // release the ownership of the request
    // std::unique_ptr<Request> request_ptr(request);
    // notify the request context that the request has finished
    // TODO: response to the client earlier
    // request->finish();
  }
}

}  // namespace llm
