#pragma once

#include <folly/MPMCQueue.h>

#include <cstdint>
#include <memory>

#include "request_context.h"
#include "request_queue.h"
#include "scheduler.h"

namespace llm {

class NaiveBatchingScheduler : public Scheduler {
 public:
  NaiveBatchingScheduler();

  // schedule a request, thread safe and non-blocking
  // may return false if the queue is full
  bool schedule(std::unique_ptr<RequestContext> request) override {
    return request_queue_.write(std::move(request));
  }

  // step the scheduler forward by one step
  // may get blocked if there are no requests to process
  void step() override;

 private:
  // a thread safe queue of requests
  folly::MPMCQueue<std::unique_ptr<RequestContext>> request_queue_;

  // a priority queue of requests, only used by the scheduling thead
  RequestPriorityQueue priority_queue_;

  // a batch of requests to be processed
  std::vector<std::unique_ptr<RequestContext>> batch_;

  // maximum number of requests in a batch
  size_t max_batch_size_ = 0;

  // maximum delay in nanoseconds before a batch is processed
  uint64_t max_batch_delay_ns_ = 0;
};

}  // namespace llm
