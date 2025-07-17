#pragma once

#include <cuda.h>

#include <cute/config.hpp>

namespace llm {

CUTE_HOST_DEVICE constexpr int clz(int x) {
  for (int i = 31; i >= 0; --i) {
    if ((1 << i) & x) {
      return int(31 - i);
    }
  }
  return int(32);
}

CUTE_HOST_DEVICE constexpr bool is_pow2(int x) { return (x & (x - 1)) == 0; }

CUTE_HOST_DEVICE constexpr int log2(int x) {
  int a = int(31 - clz(x));
  // add 1 if not a power of 2
  if (!is_pow2(x)) {
    a += 1;
  }
  return a;
}

// wrapper of PTX ex2.approx instruction, which computes 2^x
CUTE_HOST_DEVICE float exp2(float x) {
#if defined(__CUDA_ARCH__)
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
#else
  return std::exp2(x);
#endif
}

// wrapper of PTX rcp.approx instruction, which computes 1/x
CUTE_HOST_DEVICE float rcp(float x) {
#if defined(__CUDA_ARCH__)
  float y;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
#else
  return 1.0f / x;
#endif
}

// wrapper of PTX tanh.approx instruction, which computes tanh(x)
CUTE_HOST_DEVICE float tanh(float x) {
#if defined(__CUDA_ARCH__)
  float y;
  asm volatile("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
#else
  return std::tanh(x);
#endif
}

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
