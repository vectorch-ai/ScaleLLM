#include "scheduler/scheduler_config.h"

namespace llm {

SchedulerType SchedulerType::CONTINOUS_BATCHING("continous_batching");
SchedulerType SchedulerType::SPECULATIVE("speculative");

SchedulerPolicyType SchedulerPolicyType::FCFS("fcfs");
SchedulerPolicyType SchedulerPolicyType::PSA("psa");

} // namespace llm
