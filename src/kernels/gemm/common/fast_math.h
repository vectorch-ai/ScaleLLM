#pragma once

#include <cuda.h>

#include <cute/config.hpp>

namespace llm {
struct FastDivmod {
  int32_t div_ = 1;
  uint32_t mul_ = 0u;
  uint32_t shr_ = 0u;

  CUTE_HOST_DEVICE
  void reset(int div) {
    div_ = div;
    if (div_ != 1) {
      unsigned int p = 31 + log2(div_);
      unsigned m =
          unsigned(((1ull << p) + unsigned(div_) - 1) / unsigned(div_));

      mul_ = m;
      shr_ = p - 32;
    }
  }

  constexpr FastDivmod() = default;

  CUTE_HOST_DEVICE
  FastDivmod(int div) { reset(div); }

  CUTE_HOST_DEVICE
  FastDivmod& operator=(int div) {
    reset(div);
    return *this;
  }

  CUTE_HOST_DEVICE
  void divmod(int src, int& quo, int& rem) const {
    quo = div(src);
    rem = src - (quo * div_);
  }

  CUTE_HOST_DEVICE
  int div(int src) const {
#if defined(__CUDA_ARCH__)
    return (div_ != 1) ? __umulhi(src, mul_) >> shr_ : src;
#else
    return src / div_;
#endif
  }

  CUTE_HOST_DEVICE
  int mod(int src) const {
#if defined(__CUDA_ARCH__)
    return div_ != 1 ? src - (div(src) * div_) : 0;
#else
    return src % div_;
#endif
  }

  CUTE_HOST_DEVICE
  operator int() const { return div_; }
};

// operator overloads for FastDivmod
CUTE_HOST_DEVICE int operator/(int src, const FastDivmod& d) {
  return d.div(src);
}
CUTE_HOST_DEVICE int operator%(int src, const FastDivmod& d) {
  return d.mod(src);
}

}  // namespace llm
