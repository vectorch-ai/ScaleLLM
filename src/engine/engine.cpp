#include "engine.h"

#include "request/request.h"
#include "worker.h"

namespace llm {

Engine::Engine(const std::vector<torch::Device>& devices) {
  // create a worker for each device
  for (const auto& device : devices) {
    workers_.emplace_back(std::make_unique<Worker>(device));
  }

}

void Engine::forward(const std::vector<Request*>& batch) {}

}  // namespace llm
