#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace llm {
class BlockAllocator;

struct CachePlan {
  std::vector<std::pair<uint32_t, uint32_t>> swap_ins;
  std::vector<std::pair<uint32_t, uint32_t>> swap_outs;

  CachePlan(std::vector<std::pair<uint32_t, uint32_t>> _swap_ins,
            std::vector<std::pair<uint32_t, uint32_t>> _swap_outs)
      : swap_ins(std::move(_swap_ins)), swap_outs(std::move(_swap_outs)) {}
};

// CachePlanner is responsible for planning the cache memory allocation based on
// current memory consumption, memory budget as well as the memory
// requirements of requests.
class CachePlanner final {
 public:
  std::unique_ptr<CachePlan> create_plan();

  // try to schedule a request to the memory planner
  // return true if the cache memory requirements of the request can be
  // fulfilled return false otherwise
  bool try_to_schedule_request();

  // preempt a request to release memory allocation for other high priority
  // requests.
  bool preempte_request();

 private:
  // swap in: copy block from cpu to gpu
  std::vector<std::pair<uint32_t, uint32_t>> block_swap_ins_;
  // swap out: copy block from gpu to cpu
  std::vector<std::pair<uint32_t, uint32_t>> block_swap_outs_;

  // the block allocator that manages the memory blocks
  BlockAllocator* block_allocator_;
};

}  // namespace llm
