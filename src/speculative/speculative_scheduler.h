#pragma once

#include <memory>

#include "engine/engine.h"
#include "engine/llm_engine.h"
#include "request/request.h"
#include "scheduler/response_handler.h"
#include "scheduler/scheduler.h"
#include "scheduler/scheduler_config.h"
#include "scheduler/scheduler_policy.h"

namespace llm {

class BlockManager;
class LLMEngine;
class Tokenizer;
class SpeculativeScheduler final : public Scheduler {
 public:
  SpeculativeScheduler(const SchedulerConfig& config,
                       LLMEngine* llm_engine,
                       LLMEngine* ssm_engine);
  ~SpeculativeScheduler() override = default;

  bool schedule(std::unique_ptr<Request>& request) override;
  void step(const absl::Duration& timeout) override;

 private:
  void speculate_multiple_steps(std::vector<Sequence*>& sequences);
  void validate(std::vector<Sequence*>& sequences);

  SchedulerConfig config_;

  LLMEngine* llm_engine_;
  LLMEngine* ssm_engine_;

  BlockManager* llm_block_manager_;
  BlockManager* ssm_block_manager_;

  std::unique_ptr<Tokenizer> tokenizer_;

  std::unique_ptr<SchedulerPolicy> scheduler_policy_;
  std::unique_ptr<ResponseHandler> response_handler_;
};

}  // namespace llm
