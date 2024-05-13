#include "threadpool.h"

#include <thread>

#include "concurrent_queue.h"

namespace llm {
ThreadPool::ThreadPool(size_t num_threads) {
  for (size_t i = 0; i < num_threads; ++i) {
    threads_.emplace_back([this]() { internal_loop(); });
  }
}

ThreadPool::~ThreadPool() {
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
void ThreadPool::schedule(Runnable runnable) {
  if (runnable == nullptr) {
    return;
  }
  queue_.push(std::move(runnable));
}

void ThreadPool::internal_loop() {
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
