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

  // get the data pointer
  const T* data() const { return data_; }

 private:
  const T* data_ = nullptr;
  size_t size_ = 0;
};

}  // namespace llm
