#pragma once
#include <functional>
#include <thread>

#include "concurrent_queue.h"

namespace llm {

class Executor final {
 public:
  // a runnable is an object intended to be executed by the executor
  // it must be invokable with no arguments and return void.
  using Runnable = std::function<void()>;

  // constructors
  Executor() : Executor(1) {}

  // disable copy/move constructor and assignment
  Executor(const Executor&) = delete;
  Executor& operator=(const Executor&) = delete;
  Executor(Executor&&) = delete;
  Executor& operator=(Executor&&) = delete;

  explicit Executor(size_t num_threads);

  // destructor
  ~Executor();

  // schedule a runnable to be executed
  void schedule(Runnable runnable);

 private:
  void internal_loop();

  std::vector<std::thread> threads_;
  ConcurrentQueue<Runnable> queue_;
};

}  // namespace llm
