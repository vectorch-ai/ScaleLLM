#include "executor.h"

#include <functional>
#include <thread>

#include "concurrent_queue.h"

namespace llm {
Executor::Executor(size_t num_threads) {
  for (size_t i = 0; i < num_threads; ++i) {
    threads_.emplace_back([this]() { internal_loop(); });
  }
}

Executor::~Executor() {
  // push nullptr to the queue to signal threads to exit
  for (size_t i = 0; i < threads_.size(); ++i) {
    queue_.push(nullptr);
  }
  // wait for all threads to finish
  for (auto& thread : threads_) {
    thread.join();
  }
}

// schedule a runnable to be executed
void Executor::schedule(Runnable runnable) {
  if (runnable == nullptr) {
    return;
  }
  queue_.push(std::move(runnable));
}

void Executor::internal_loop() {
  while (true) {
    Runnable runnable = queue_.pop();
    if (runnable == nullptr) {
      // nullptr is a signal to exit
      break;
    }
    runnable();
  }
}

}  // namespace llm
