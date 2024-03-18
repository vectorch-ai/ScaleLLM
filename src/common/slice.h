#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace llm {

template <typename T>
class Slice final {
 public:
  Slice() = default;

  Slice(const T* data, size_t size) : data_(data), size_(size) {}

  Slice(const std::vector<T>& data, size_t stat_pos = 0)
      : data_(data.data() + stat_pos), size_(data.size() - stat_pos) {}

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
  Slice<T> sub(size_t start) const {
    return Slice<T>(data_ + start, size_ - start);
  }

  // align the slice to allignment
  Slice<T> align_to(size_t alignment) const {
    const size_t alligned_size = size_ / alignment * alignment;
    return Slice<T>(data_, alligned_size);
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
