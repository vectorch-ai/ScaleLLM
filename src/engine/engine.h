#pragma once

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

  // step the engine forward by one step with the batch
  virtual void forward(const std::vector<Request*>& batch) {}
};

}  // namespace llm
