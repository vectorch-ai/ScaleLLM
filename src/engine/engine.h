#pragma once

#include "memory/cache_planner.h"
#include "request/request.h"

namespace llm {

// The Large Language Model (LLM) engine is a model runner designed to execute
// inference procedures incrementally using batches of requests. It comprises
// three critical components: a model, a tokenizer, and a resource manager.
// The inference process is primarily divided into two stages: 'prefill' and
// 'generate'.
// * 'Prefill': This is the more costly phase, as it involves processing a
// new prompt and generating the entire initial attention matrix.
// * 'Generate': In this phase, subsequent tokens are generated using the
// previously cached attention matrix.
// A single batch may contain requests from various stages of the inference
// process. The engine must be adept at handling these diverse requests,
// ensuring optimal resource management.

class Engine {
 public:
  virtual ~Engine() = default;

  std::unique_ptr<CachePlanner> create_cache_planner();

  // // prepare the batch for execution, the task may include:
  // // 1. tokenize the input text for new requests
  // // 2. allocate resource for new requests
  // // 3. resource management for preempted requests
  // virtual void prepare_batch(const std::vector<Request*>& batch,
  //                            const CachePlan& cache_plan) {}

  // step the engine forward by one step with the batch
  virtual void forward(const std::vector<Request*>& batch,
                       const CachePlan* cache_plan) {}
};

}  // namespace llm
