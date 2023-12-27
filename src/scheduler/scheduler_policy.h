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

class BlockManager;
class ResponseHandler;
class Sequence; 
// First come first serve scheduler policy
class FCFSSchedulerPolicy final : public SchedulerPolicy {
 public:
  explicit FCFSSchedulerPolicy(ResponseHandler* response_handler,
                               BlockManager* block_manager);
  ~FCFSSchedulerPolicy() override;

  bool try_emplace(std::unique_ptr<Request>& request) override;
  void schedule() override;

 private:
  ResponseHandler* response_handler_;
  BlockManager* block_manager_;

  folly::MPMCQueue<Request*> waiting_queue_;
  std::vector<Request*> blocking_queue_;
  std::vector<Request*> running_queue_;

  std::vector<Sequence*> running_batch_;
};

}  // namespace llm
