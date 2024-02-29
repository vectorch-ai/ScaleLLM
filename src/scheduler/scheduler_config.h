#pragma once
#include <string>

namespace llm {
namespace {
constexpr uint64_t kSpeculativeSteps = 10; 
}

class SchedulerType {
 public:
  SchedulerType(const std::string& type) : type_(type) {}

  static SchedulerType CONTINOUS_BATCHING;
  static SchedulerType SPECULATIVE;

 private:
  std::string type_;
};

class SchedulerPolicyType {
 public:
  SchedulerPolicyType(const std::string& type) : type_(type) {}

  static SchedulerPolicyType FCFS;
  static SchedulerPolicyType PSA;

 private:
  std::string type_;
};

struct SchedulerConfig {
  SchedulerConfig(SchedulerType type, SchedulerPolicyType policy_type,
      uint64_t speculative_steps) : type_(type), policy_type_(policy_type),
          speculative_steps_(speculative_steps) {}

  SchedulerConfig(SchedulerType type, SchedulerPolicyType policy_type)
      : type_(type), policy_type_(policy_type) {}

  SchedulerType type_ = SchedulerType::CONTINOUS_BATCHING;
  SchedulerPolicyType policy_type_ = SchedulerPolicyType::PSA;
  
  // speculative configuration
  const uint64_t speculative_steps_ = kSpeculativeSteps;
};

}  // namespace llm
