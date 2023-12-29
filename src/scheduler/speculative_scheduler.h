#pragma once

#include <cstdint>
#include <memory>

#include "engine/engine.h"
#include "request/request.h"
#include "scheduler/scheduler.h"
#include "scheduler/scheduler_config.h"

namespace llm {

class BlockManager;
class Engine;
class ResponseHandler;
class SchedulerPolicy;
class Tokenizer;
class SpeculativeScheduler final : public Scheduler {
 public:
  SpeculativeScheduler(const SchedulerConfig& config,
                       Engine* llm_engine,
                       Engine* ssm_engine);
  ~SpeculativeScheduler() override;

  bool schedule(std::unique_ptr<Request>& request) override;
  void step(const absl::Duration& timeout) override;

 private:
  void speculate_multiple_steps(std::vector<Sequence*>& sequences);
  OutputParameters validate(std::vector<Sequence*>& sequences);
 
 private:
  SchedulerConfig config_;

  Engine* llm_engine_;
  Engine* ssm_engine_;

  BlockManager* llm_block_manager_;
  BlockManager* ssm_block_manager_; 

  std::unique_ptr<Tokenizer> tokenizer_;

  SchedulerPolicy* scheduler_policy_;
  ResponseHandler* response_handler_;
};

}  // namespace llm
