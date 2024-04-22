#pragma once

#include "scheduler/scheduler_config.h"
// #include "scheduler/speculative_scheduler.h"

namespace llm {

class Engine;
class Scheduler;
class SchedulerFactory {
 public:
  static Scheduler* Create(const SchedulerConfig& config,
                           Engine* llm_engine,
                           Engine* ssm_engine) {
    // static SpeculativeScheduler scheduler(config, llm_engine, ssm_engine);
    // return &scheduler;
    // TODO: implement this function
    return nullptr;
  }
};

}  // namespace llm
