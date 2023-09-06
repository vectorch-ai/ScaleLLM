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

constexpr uint64_t kStepSleepTimeMs = 10;

ContinuousBatchingScheduler::ContinuousBatchingScheduler()
    : request_queue_(kRequestQueueSize) {}

ContinuousBatchingScheduler::~ContinuousBatchingScheduler() {
  // release all requests in the queue
  while (!request_queue_.isEmpty()) {
    Request* request = nullptr;
    request_queue_.read(request);
    std::unique_ptr<Request> request_ptr(request);
  }

  // release all requests in the priority queue
  while (!priority_queue_.empty()) {
    Request* request = priority_queue_.top();
    priority_queue_.pop();
    std::unique_ptr<Request> request_ptr(request);
  }

  // release all requests in the batch
  for (Request* request : batch_) {
    std::unique_ptr<Request> request_ptr(request);
  }
  batch_.clear();
}

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

void ContinuousBatchingScheduler::create_batch() {
  // propogate new requests to priority_queue_
  while (!request_queue_.isEmpty()) {
    Request* request = nullptr;
    // read from request then then push to priority queue
    request_queue_.read(request);
    CHECK(request != nullptr);
    priority_queue_.push(request);
  }

  // fast paths
  // no new requests
  if (priority_queue_.empty()) {
    return;
  }
  // requests in current batch all have precedence over new requests
  if (!batch_.empty() &&
      RequestPtrLess()(batch_.back(), priority_queue_.top())) {
    // TODO: need to check if we can schedule new requests
    return;
  }

  // add requests in current batch back to the priority queue
  std::unordered_map<Request*, size_t> request_to_idx;
  for (size_t i = 0; i < batch_.size(); ++i) {
    Request* request = batch_[i];
    // TODO: skip finished requests
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
      Request* request_to_preempt = batch_[--end_idx];
      block_manager_->release_slots_for_request(request_to_preempt);
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
  batch_ = std::move(new_batch);
}

// step the scheduler forward by one step
// may get blocked if there are no requests to process
void ContinuousBatchingScheduler::step(const absl::Duration& timeout) {
  // get a new batch of requests
  const auto deadline = absl::Now() + timeout;
  while (true) {
    create_batch();
    if (!batch_.empty()) {
      // find one batch of requests to process
      break;
    }
    const auto now = absl::Now();
    if (now > deadline) {
      // no requests to process
      return;
    }
    // wait for new requests to arrive
    const auto time_to_sleep =
        std::min(absl::Milliseconds(kStepSleepTimeMs), deadline - now);
    absl::SleepFor(time_to_sleep);
  }

  CHECK(!batch_.empty());
  auto output_parameters = engine_->execute_model(batch_);

  // TODO: process finished requests
  for (auto& request : batch_) {
    // release the ownership of the request
    // std::unique_ptr<Request> request_ptr(request);
    // notify the request context that the request has finished
    // TODO: response to the client earlier
    // request->finish();

    // update batch status, next token ids.
    // 1> response to client if stream is enabled
    // 2> remove request from batch if it is finished
  }
}

}  // namespace llm
