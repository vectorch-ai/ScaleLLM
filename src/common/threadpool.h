#pragma once
#include <folly/Function.h>
#include <thread>

#include "concurrent_queue.h"

namespace llm {

class ThreadPool final {
 public:
  // a runnable is an object intended to be executed by the threadpool
  // it must be invokable with no arguments and return void.
  using Runnable = folly::Function<void()>;

  // constructors
  ThreadPool() : ThreadPool(1) {}

  // disable copy/move constructor and assignment
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;
  ThreadPool(ThreadPool&&) = delete;
  ThreadPool& operator=(ThreadPool&&) = delete;

  explicit ThreadPool(size_t num_threads);

  // destructor
  ~ThreadPool();

  // schedule a runnable to be executed
  void schedule(Runnable runnable);

 private:
  void internal_loop();

  std::vector<std::thread> threads_;
  ConcurrentQueue<Runnable> queue_;
};

}  // namespace llm
