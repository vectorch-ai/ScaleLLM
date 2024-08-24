#pragma once

#include <algorithm>
#include <type_traits>

#include "macros.h"

namespace llm {

template <typename T, REQUIRES(std::is_integral_v<T>)>
class IntegerAange {
 public:
  class iterator;
  IntegerAange(T begin, T end) : begin_(begin), end_(end) {}

  iterator begin() const { return begin_; }

  iterator end() const { return end_; }

  class iterator {
   public:
    explicit iterator(T value) : value_(value) {}

    T operator*() const { return value_; }

    const T* operator->() const { return &value_; }

    iterator& operator++() {
      ++value_;
      return *this;
    }

    bool operator==(const iterator& other) const {
      return value_ == other.value_;
    }

    bool operator!=(const iterator& other) const {
      return value_ != other.value_;
    }

   private:
    T value_;
  };

 private:
  iterator begin_;
  iterator end_;
};

// create a range for half-open interval [0, end)
template <typename T, REQUIRES(std::is_integral_v<T>)>
IntegerAange<T> range(T end) {
  return {T(), end};
}

// create a range for half-open interval [begin, end)
// return an empty range if begin >= end
template <typename T, REQUIRES(std::is_integral_v<T>)>
IntegerAange<T> range(T begin, T end) {
  return {begin, std::max(begin, end)};
}

}  // namespace llm