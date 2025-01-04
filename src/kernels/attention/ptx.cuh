#pragma once

#include <cuda.h>
#include <cute/config.hpp>

namespace llm::ptx {

// wrapper of PTX ex2.approx instruction, which computes 2^x
CUTE_DEVICE float exp2(float x) {
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// wrapper of PTX rcp.approx instruction, which computes 1/x
CUTE_DEVICE float rcp(float x) {
  float y;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// wrapper of PTX tanh.approx instruction, which computes tanh(x)
CUTE_DEVICE float tanh(float x) {
  float y;
  asm volatile("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

}  // namespace llm::ptx