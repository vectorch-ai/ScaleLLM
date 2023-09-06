#include "naive_batching_scheduler.h"

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

NaiveBatchingScheduler::NaiveBatchingScheduler()
    : request_queue_(kRequestQueueSize) {}

bool NaiveBatchingScheduler::schedule(std::unique_ptr<Request>& request) {
  CHECK(request != nullptr);
  if (request_queue_.write(request.get())) {
    // take over the ownership of the request
    request.release();
    return true;
  }
  // queue is full
  return false;
}

std::vector<Request*> NaiveBatchingScheduler::get_batch(
    const absl::Duration& max_batch_delay) {
  // propogate requests from request_queue_ to priority_queue_
  while (!request_queue_.isEmpty()) {
    Request* request = nullptr;
    // read from request then then push to priority queue
    request_queue_.read(request);
    CHECK(request != nullptr);
    priority_queue_.push(request);
  }

  std::vector<Request*> batch;
  // get a batch of requests from the priority queue
  absl::Time deadline = absl::Now() + max_batch_delay;
  while (true) {
    // get all available requests from the priority queue
    while (!priority_queue_.empty()) {
      Request* candidate = priority_queue_.top();
      if (!block_manager_->allocate_slots_for_request(candidate)) {
        // no more slots available, return the batch
        return batch;
      }

      batch.push_back(candidate);
      priority_queue_.pop();
      if (batch.size() >= kMaxBatchSize) {
        // batch is full, return the batch
        return batch;
      }
    }

    // wait for more requests to arrive if batch delay has not been exceeded
    const absl::Time now = absl::Now();
    if (now > deadline) {
      return batch;
    }
    const absl::Duration wait_time =
        std::min(deadline - now, absl::Milliseconds(2));
    absl::SleepFor(wait_time);
  }
  // should not reach here
  return batch;
}

bool NaiveBatchingScheduler::is_batch_finished() {
  for (auto& request : batch_) {
    if (!request->is_finished()) {
      return false;
    }
  }
  return true;
}

// step the scheduler forward by one step
// may get blocked if there are no requests to process
void NaiveBatchingScheduler::step(const absl::Duration& /*timeout*/) {
  // check if all requests in the batch have been fulfilled
  if (!is_batch_finished()) {
    // engine_->execute_model(batch_);
    return;
  }

  // process finished requests
  for (auto& request : batch_) {
    // release the ownership of the request
    std::unique_ptr<Request> request_ptr(request);
    // notify the request context that the request has finished
    // TODO: response to the client earlier
    request->finish();
  }

  // get a new batch of requests
  batch_ = get_batch(absl::Milliseconds(max_batch_delay_ns_));
  // engine_->execute_model(batch_);
}

}  // namespace llm
