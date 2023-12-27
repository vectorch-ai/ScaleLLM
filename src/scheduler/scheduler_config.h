#pragma once
#include <string>

namespace llm {

constexpr uint64_t kSpeculativeSteps = 10; 

class SchedulerType {
 public:
  SchedulerType(const std::string& type) : type_(type) {}

  static SchedulerType CONTINOUS_BATCHING;
  static SchedulerType SPECULATIVE;

 private:
  std::string type_;
};

SchedulerType SchedulerType::CONTINOUS_BATCHING("continous_batching");
SchedulerType SchedulerType::SPECULATIVE("speculative");

struct SchedulerConfig {
  SchedulerType type = SchedulerType::CONTINOUS_BATCHING;
  
  // speculative configuration
  const uint64_t speculative_steps = kSpeculativeSteps;
};

}  // namespace llm
