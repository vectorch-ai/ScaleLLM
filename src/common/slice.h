#pragma once

#include <glog/logging.h>

#include <vector>

namespace llm {

template <typename T>
class Slice final {
 public:
  Slice() = default;

  Slice(const T* data, size_t size) : data_(data), size_(size) {}

  explicit Slice(const std::vector<T>& data)
      : data_(data.data()), size_(data.size()) {}

  Slice(const std::vector<T>& data, size_t size)
      : data_(data.data()), size_(size) {
    CHECK(size <= data.size());
  }

  // iterator for the slice
  const T* begin() const { return data_; }
  const T* end() const { return data_ + size_; }

  // get the size of the slice
  size_t size() const { return size_; }

  // check if the slice is empty
  bool empty() const { return size_ == 0; }

  // get the data pointer
  const T* data() const { return data_; }

  // index operator
  const T& operator[](size_t i) const { return data_[i]; }

  // get a sub slice
  Slice<T> slice(size_t start) const {
    CHECK(start <= size_);
    return Slice<T>(data_ + start, size_ - start);
  }

  Slice<T> slice(size_t start, size_t end) const {
    CHECK(start <= end && end <= size_);
    return Slice<T>(data_ + start, end - start);
  }

  // convert to vector
  std::vector<T> to_vector() const {
    return std::vector<T>(data_, data_ + size_);
  }

 private:
  const T* data_ = nullptr;
  size_t size_ = 0;
};

}  // namespace llm