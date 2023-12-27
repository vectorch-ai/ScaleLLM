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

class SchedulerPolicyType {
 public:
  SchedulerPolicyType(const std::string& type) : type_(type) {}

  static SchedulerPolicyType FCFS;
  static SchedulerPolicyType PSA;

 private:
  std::string type_;
};

SchedulerPolicyType SchedulerPolicyType::FCFS("fcfs");
SchedulerPolicyType SchedulerPolicyType::PSA("psa");

struct SchedulerConfig {
  SchedulerType type = SchedulerType::CONTINOUS_BATCHING;
  SchedulerPolicyType policy_type = SchedulerPolicyType::PSA;
  
  // speculative configuration
  const uint64_t speculative_steps = kSpeculativeSteps;
};

}  // namespace llm
