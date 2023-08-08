#pragma once

#include <absl/time/time.h>
#include <folly/MPMCQueue.h>

#include <cstdint>
#include <memory>
#include <queue>

#include "engine/engine.h"
#include "memory/block_manager.h"
#include "request/request.h"
#include "scheduler.h"

namespace llm {

// An implementation of Naive Batching (Batch Requests by Time): In this
// strategy, requests are buffered into batches before being sent for inference.
// The batch size remains constant until the inference process completes.
class NaiveBatchingScheduler : public Scheduler {
 public:
  NaiveBatchingScheduler();

  // schedule a request, thread safe and non-blocking
  // may return false if the queue is full
  bool schedule(std::unique_ptr<Request>& request) override;

  // step the scheduler forward by one step
  // may get blocked if there are no requests to process
  void step(const absl::Duration& timeout) override;

 private:
  // get a batch of requests from the priority queue
  std::vector<Request*> get_batch(const absl::Duration& max_batch_delay);

  // all requests in the batch are finished
  bool is_batch_finished();

  // a thread safe queue of requests, bounded by kRequestQueueSize
  // the schedule owns the requests and manages their lifetimes.
  folly::MPMCQueue<Request*> request_queue_;

  // Requests with HIGH priority are processed first, followed by MEDIUM
  // priority requests, and finally LOW priority requests. Within each priority
  // level, requests are handled on First-Come-First-Served (FCFS) basis.
  using MinHeap =
      std::priority_queue<Request*, std::vector<Request*>, RequestPtrLess>;
  MinHeap priority_queue_;

  // a batch of requests to be processed
  std::vector<Request*> batch_;

  // maximum number of requests in a batch
  size_t max_batch_size_ = 0;

  // maximum delay in nanoseconds before a batch is processed
  uint64_t max_batch_delay_ns_ = 0;

  std::unique_ptr<BlockManager> block_manager_;

  // the engine to run the batch
  std::unique_ptr<Engine> engine_;
};

}  // namespace llm
