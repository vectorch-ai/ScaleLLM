#pragma once

#include <absl/synchronization/mutex.h>

#include <queue>

#if __has_attribute(guarded_by)
#define GUARDED_BY(x) __attribute__((guarded_by(x)))
#else
#define GUARDED_BY(x)
#endif

namespace llm {

// a simple thread-safe queue that supports multiple producers and multiple
// consumers concurrently the queue is implemented as a queue with condition
// variable and mutex lock
template <typename T>
class ConcurrentQueue {
 public:
  // constructor
  ConcurrentQueue() = default;

  explicit ConcurrentQueue(size_t capacity) : capacity_(capacity) {}

  // destructor
  ~ConcurrentQueue() = default;

  // push an element to the queue
  void push(T value) {
    absl::MutexLock lock(&mutex_);
    if (capacity_ > 0) {
      auto not_full = [this]() { return queue_.size() < capacity_; };
      mutex_.Await(absl::Condition(&not_full));
    }
    queue_.push(std::move(value));
  }

  template <typename... Args>
  void emplace(Args&&... args) {
    absl::MutexLock lock(&mutex_);
    if (capacity_ > 0) {
      auto not_full = [this]() { return queue_.size() < capacity_; };
      mutex_.Await(absl::Condition(&not_full));
    }
    queue_.emplace(std::forward<Args>(args)...);
  }

  // pop an element from the queue, block if the queue is empty
  T pop() {
    absl::MutexLock lock(&mutex_);

    auto not_empty = [this]() { return !queue_.empty(); };
    mutex_.Await(absl::Condition(&not_empty));

    T value = std::move(queue_.front());
    queue_.pop();
    return value;
  }

  // return the size of the queue
  size_t size() {
    absl::MutexLock lock(&mutex_);
    return queue_.size();
  }

  // return true if the queue is empty
  bool empty() {
    absl::MutexLock lock(&mutex_);
    return queue_.empty();
  }

 private:
  // the underlying queue
  std::queue<T> queue_ GUARDED_BY(mutex_);
  // mutex lock for the queue
  absl::Mutex mutex_;

  // maximum capacity of the queue, 0 means no limit.
  // when the queue is full, push will block
  size_t capacity_ = 0;
};
}  // namespace llm
