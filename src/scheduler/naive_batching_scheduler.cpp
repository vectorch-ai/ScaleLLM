#include "naive_batching_scheduler.h"

#include <folly/MPMCQueue.h>

#include <cstdint>
#include <memory>

#include "request/request_context.h"
#include "request_queue.h"

namespace llm {
constexpr size_t kRequestQueueSize = 100000;

NaiveBatchingScheduler::NaiveBatchingScheduler()
    : request_queue_(kRequestQueueSize) {}

// step the scheduler forward by one step
// may get blocked if there are no requests to process
void NaiveBatchingScheduler::step() {
  if (!batch_.empty()) {
    // a batch is being processed, continue processing it until it is done
    // llm->forward(batch)

    // check if all requests in the batch have been fulfilled
    // if so, release resources
    // batch_.clear();
    return;
  }

  // pull queued requests into the priority queue
  while (!request_queue_.isEmpty()) {
    std::unique_ptr<RequestContext> request;
    request_queue_.read(request);
    priority_queue_.enqueue(std::move(request));
  }

  // TODO: remove requests that have timed out

  // check if we found a preferred batch size and the batch delay has been
  // exceeded if so, process the batch

  // prepare the prefill input for the batch

  // process the batch by calling llm->forward(batch)

  // release resources
}

}  // namespace llm
