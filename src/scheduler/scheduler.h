#pragma once

#include <memory>
#include <string>
#include <absl/time/time.h>

#include "request/request.h"

namespace llm {

// A scheduler is a crucial component responsible for managing incoming requests
// and orchestrating their processing. In the context of LLM scheduling, the
// choice between utilizing GPU or CPU for computation significantly impacts
// performance. GPUs excel in massively-parallel computation, providing
// remarkable processing power. However, GPUs prioritize throughput over
// latency, necessitating request batching to attain high throughput.

// To achieve this, the scheduler plays a vital role in batching requests. There
// are two main batching strategies to consider:

// A: Naive Batching (Batch Requests by Time): In this strategy, requests are
// buffered into batches before being sent for inference. The batch size remains
// constant until the inference process completes. However, due to the iterative
// nature of inference, some requests may finish earlier than others, causing
// resource inefficiencies. It becomes challenging to release resources and
// schedule new requests promptly, leading to suboptimal GPU utilization.

// B: Continuous Batching (Batch Requests Dynamically): In contrast, continuous
// batching dynamically adjusts the batch size as inference progresses. This
// approach requires a more intricate implementation but offers higher GPU
// utilization. As the inference proceeds, the scheduler continuously optimizes
// the batch size to accommodate the varying computational loads, resulting in
// improved overall efficiency.
class Scheduler {
 public:
  virtual ~Scheduler() = default;

  // schedule a request. thread safe
  // return true if the request is scheduled successfully.
  // false otherwise and the ownership of the request is not transferred.
  virtual bool schedule(std::unique_ptr<Request>& request) = 0;

  // step the scheduler forward by one step
  // may get blocked if there are no requests to process
  // not thread safe
  virtual void step(const absl::Duration& timeout) = 0;
};

}  // namespace llm
