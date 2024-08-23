#pragma once

#include <glog/logging.h>

#include <algorithm>
#include <numeric>
#include <vector>

namespace llm {

namespace detail {
inline size_t size(const std::vector<size_t>& shape) {
  if (shape.empty()) {
    return 0;
  }
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
}

inline std::vector<size_t> row_major_stride(const std::vector<size_t>& shape) {
  std::vector<size_t> stride(shape.size());
  size_t stride_val = 1;
  for (int32_t i = shape.size() - 1; i >= 0; --i) {
    stride[i] = stride_val;
    stride_val *= shape[i];
  }
  return stride;
}

}  // namespace detail

using Coord = std::vector<size_t>;
template <class... Ts>
Coord make_coord(const Ts&... t) {
  return {static_cast<size_t>(t)...};
}

template <typename T>
class Array {
 public:
  Array(T* data,
        const std::vector<size_t>& shape,
        const std::vector<size_t>& stride)
      : data_(data),
        shape_(shape),
        stride_(stride),
        size_(detail::size(shape)) {
    CHECK_EQ(shape.size(), stride.size());
  }

  Array(T* data, const std::vector<size_t>& shape)
      : Array(data, shape, detail::row_major_stride(shape)) {}

  // indexing with offset
  T& operator[](size_t offset) {
    CHECK_LT(offset, size_);
    return data_[offset];
  }

  T operator[](size_t offset) const {
    CHECK_LT(offset, size_);
    return data_[offset];
  }

  // indexing with multi-dimensional coordinates
  T& operator[](const Coord& coord) {
    return operator[](coord_to_offset(coord));
  }

  T operator[](const Coord& coord) const {
    return operator[](coord_to_offset(coord));
  }

  // indexing with offset
  T& operator()(size_t offset) {
    CHECK_LT(offset, size_);
    return data_[offset];
  }

  T operator()(size_t offset) const {
    CHECK_LT(offset, size_);
    return data_[offset];
  }

  // indexing with multi-dimensional coordinates
  T& operator()(const Coord& coord) {
    return operator()(coord_to_offset(coord));
  }

  T operator()(const Coord& coord) const {
    return operator()(coord_to_offset(coord));
  }

  // convenience function for multi-dimensional indexing
  template <class Coord0, class Coord1, class... Coords>
  T& operator()(const Coord0& coord0,
                const Coord1& coord1,
                const Coords&... coords) {
    return operator()(make_coord(coord0, coord1, coords...));
  }

  template <class Coord0, class Coord1, class... Coords>
  T operator()(const Coord0& coord0,
               const Coord1& coord1,
               const Coords&... coords) const {
    return operator()(make_coord(coord0, coord1, coords...));
  }

  size_t size() const { return size_; }

  const std::vector<size_t>& shape() const { return shape_; }

  const std::vector<size_t>& stride() const { return stride_; }

  T* data() { return data_; }

  const T* data() const { return data_; }

 private:
  size_t coord_to_offset(const std::vector<size_t>& coord) const {
    CHECK_EQ(coord.size(), shape_.size());
    int64_t offset = 0;
    for (int i = 0; i < coord.size(); ++i) {
      CHECK_LT(coord[i], shape_[i]);
      offset += coord[i] * stride_[i];
    }
    return offset;
  }

  // data pointer
  T* data_;

  // shape and stride
  std::vector<size_t> shape_;
  std::vector<size_t> stride_;

  // domain size
  size_t size_ = 0;
};

}  // namespace llm