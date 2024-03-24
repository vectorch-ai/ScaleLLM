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

// TODO: add schedule config to control the max number of tokens per batch, max
// number of seqs per batch and the time out value.
class ContinuousBatchingScheduler final : public Scheduler {
 public:
  ContinuousBatchingScheduler(Engine* engine);

  ~ContinuousBatchingScheduler();

  // schedule a request, thread safe and non-blocking
  // may return false if the queue is full
  bool schedule(std::unique_ptr<Request>& request) override;

  // step the scheduler forward by one step
  // may get blocked if there are no requests to process
  void step(const absl::Duration& timeout) override;

 private:
  // get a batch of requests from the priority queue
  void build_sequence_batch();

  void on_request_finish(Request* request);

  void on_sequence_stream(Sequence* seq);

  // allocate blocks for a sequence, honoring the tokens budget.
  // * for prefill sequence, the allocated_tokens will be within
  // [1, num_prompt_tokens - num_tokens_in_kv_cache].
  // * for decode sequence, the actual_tokens usually would be 1 or K for
  // speculative decoding.
  // returns false if no blocks can be allocated.
  bool allocate_blocks_for(Sequence* sequence,
                           size_t token_budget,
                           size_t* actual_tokens);

  // the engine to run the batch
  Engine* engine_;

  // the block manager to manage the cache blocks
  BlockManager* block_manager_;

  // tokenizer
  std::unique_ptr<Tokenizer> tokenizer_;

  // a thread safe queue of requests, bounded by kRequestQueueSize
  // the schedule owns the requests and manages their lifetimes.
  folly::MPMCQueue<Request*> request_queue_;

  // Requests with HIGH priority are processed first, followed by MEDIUM
  // priority requests, and finally LOW priority requests. Within each priority
  // level, requests are handled on First-Come-First-Served (FCFS) basis.
  using MinHeap =
      std::priority_queue<Request*, std::vector<Request*>, RequestPtrGreater>;
  MinHeap priority_queue_;

  // a batch of requests to be processed, sorted by priority from high to low.
  std::vector<Request*> requests_batch_;

  // a batch of sequences to be processed, sorted by priority from high to low.
  Batch sequences_batch_;

  // preemptable requests that hold cache slots, sorted by priority from high to
  // low.
  std::deque<Request*> preemptable_candidates_;

  // the threadpool to handle responses
  ThreadPool response_threadpool_;
};

}  // namespace llm
