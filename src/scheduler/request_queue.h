#pragma once

#include <cstdint>
#include <memory>
#include <queue>

#include "request_context.h"

namespace llm {

// A request queue is a data structure that holds requests and facilitates the
// prioritized enqueueing and dequeueing of these requests. There are three
// priority levels: HIGH, MEDIUM, and LOW. Requests with HIGH priority are
// processed first, followed by MEDIUM priority requests, and finally LOW
// priority requests. Within each priority level, requests are handled on
// First-Come-First-Served (FCFS) basis.
//
// [Starvation] : Please note the low priority requests may never be processed
// if there are always high priority requests. not thread safe.
class RequestPriorityQueue final {
 public:
  enum class Priority { HIGH = 0, MEDIUM, LOW };

  RequestPriorityQueue() = default;

  ~RequestPriorityQueue();

  // enqueue a request
  void enqueue(std::unique_ptr<RequestContext> request,
               Priority priority = Priority::MEDIUM);

  // dequeue a request
  std::unique_ptr<RequestContext> dequeue();

  // peek at the next request to be processed
  RequestContext* front();

  // return the number of requests in the queue
  size_t size() const { return size_; }

 private:
  // min heap that puts the request with the lowest scheduled time at the top
  using MinHeap = std::priority_queue<RequestContext*,
                                      std::vector<RequestContext*>,
                                      RequestContextPtrLess>;

  // queues for each priority level
  std::vector<MinHeap> priority_queues_ = std::vector<MinHeap>(3);

  // number of requests in the queue
  size_t size_ = 0;
};

}  // namespace llm
