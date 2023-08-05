#include "request_queue.h"

#include "request/request_context.h"

namespace llm {

RequestPriorityQueue::~RequestPriorityQueue() {
  // free all requests in queue
  for (auto& queue : priority_queues_) {
    while (!queue.empty()) {
      std::unique_ptr<RequestContext>{queue.top()};
      queue.pop();
    }
  }
}

// enqueue a request
void RequestPriorityQueue::enqueue(std::unique_ptr<RequestContext> request,
                                   Priority priority) {
  if (request == nullptr) {
    return;
  }
  ++size_;
  // take over ownership of the request
  priority_queues_[static_cast<size_t>(priority)].push(request.release());
}

// dequeue a request
std::unique_ptr<RequestContext> RequestPriorityQueue::dequeue() {
  for (auto& queue : priority_queues_) {
    if (!queue.empty()) {
      // unique_ptr will take ownership of the request
      std::unique_ptr<RequestContext> request(queue.top());
      queue.pop();
      --size_;
      return request;
    }
  }
  return nullptr;
}

RequestContext* RequestPriorityQueue::front() {
  for (auto& queue : priority_queues_) {
    if (!queue.empty()) {
      return queue.top();
    }
  }
  return nullptr;
}

}  // namespace llm
