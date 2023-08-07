#include "cache_planner.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "block_allocator.h"

namespace llm {

std::unique_ptr<CachePlan> CachePlanner::create_plan() {
  return std::make_unique<CachePlan>(std::move(block_swap_ins_),
                                     std::move(block_swap_outs_));
}

// try to schedule a request to the memory planner
// return true if the cache memory requirements of the request can be
// fulfilled return false otherwise
bool CachePlanner::try_to_schedule_request(Request* request) {
  // TODO: allocate blocks based on the request requirements
  return false;
}

// preempt a request to release memory allocation for other high priority
// requests.
bool CachePlanner::preempte_request(Request* request) {
  // TODO: release blocks based on the request cache allocation
  return false;
}

}  // namespace llm
