#pragma once

#include <glog/logging.h>

#include <algorithm>
#include <vector>

namespace llm {

template <typename T>
class Array {
 public:
  Array(T* data, const std::vector<size_t>& sizes)
      : data_(data), sizes_(sizes) {
    // by default, use row-major layout
    strides_.resize(sizes.size());
    strides_.back() = 1;
    for (int i = sizes.size() - 2; i >= 0; --i) {
      strides_[i] = strides_[i + 1] * sizes[i + 1];
    }
  }

  T& operator[](const std::vector<size_t>& indices) {
    CHECK_EQ(indices.size(), sizes_.size());
    int64_t offset = 0;
    for (int i = 0; i < indices.size(); ++i) {
      CHECK_LT(indices[i], sizes_[i]);
      offset += indices[i] * strides_[i];
    }
    return data_[offset];
  }

  T& operator()(const std::vector<size_t>& indices) {
    return operator[](indices);
  }

 private:
  T* data_;
  std::vector<size_t> sizes_;
  std::vector<size_t> strides_;
};

}  // namespace llm