#pragma once

#include <absl/time/time.h>
#include <folly/MPMCQueue.h>

#include <cstdint>
#include <memory>
#include <queue>

namespace llm {

class Request;
class SchedulerPolicy {
 public:
  virtual ~SchedulerPolicy() = default;

  virtual bool try_emplace(std::unique_ptr<Request>& request) = 0;
  virtual void schedule() = 0;
};

class Sequence; 
// First come first serve scheduler policy
class FCFSSchedulerPolicy : public SchedulerPolicy {
 public:
  FCFSSchedulerPolicy() = default;
  ~FCFSSchedulerPolicy() override;

  bool try_emplace(std::unique_ptr<Request>& request) override;
  void schedule() override;

 private:
  // a thread safe queue of requests, bounded by kRequestQueueSize
  // the schedule owns the requests and manages their lifetimes.
  folly::MPMCQueue<Request*> waiting_queue_;

  // blocking request queue
  std::vector<Request*> blocking_queue_;

  // running request queue
  std::vector<Request*> running_queue_;

  // a batch of sequence to be processed.
  std::vector<Sequence*> running_batch_;
};

}  // namespace llm
