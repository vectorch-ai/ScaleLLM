#pragma once

#include <cstdint>
#include <memory>

#include "request/request.h"
#include "scheduler.h"

namespace llm {

class BlockManager;
class Engine;
class ResponseHandler;
class SchedulerPolicy;
class Tokenizer;
class SpeculativeScheduler final : public Scheduler {
 public:
  SpeculativeScheduler(Engine* llm_engine, Engine* ssm_engine);

  ~SpeculativeScheduler() override;

  // schedule a request, thread safe and non-blocking
  // may return false if the queue is full
  bool schedule(std::unique_ptr<Request>& request) override;

  // step the scheduler forward by one step
  // may get blocked if there are no requests to process
  void step(const absl::Duration& timeout) override;
 
 private:
  Engine* ssm_engine_;
  Engine* llm_engine_;

  BlockManager* llm_block_manager_;
  BlockManager* ssm_block_manager_; 

  std::unique_ptr<Tokenizer> tokenizer_;

  SchedulerPolicy* scheduler_policy_;
  ResponseHandler* response_handler_;
};

}  // namespace llm
