#pragma once

#include <cute/int_tuple.hpp>
#include <cute/layout.hpp>

#include "sm80_mla_launch.cuh"
#include "static_dispatch.h"

namespace llm {

#define DISPATCH_HEAD_DIM_(HEAD_DIM_V, HEAD_DIM_NAME, ...) \
  [&] {                                                    \
    if (HEAD_DIM_V == 128) {                               \
      constexpr static int HEAD_DIM_NAME = 128;            \
      return __VA_ARGS__();                                \
    } else if (HEAD_DIM_V == 256) {                        \
      constexpr static int HEAD_DIM_NAME = 256;            \
      return __VA_ARGS__();                                \
    } else if (HEAD_DIM_V == 512) {                        \
      constexpr static int HEAD_DIM_NAME = 512;            \
      return __VA_ARGS__();                                \
    } else {                                               \
      assert(false);                                       \
    }                                                      \
  }()

#define DISPATCH_ROPE_HEAD_DIM_(ROPE_HEAD_DIM_V, ROPE_HEAD_DIM_NAME, ...) \
  [&] {                                                                   \
    if (ROPE_HEAD_DIM_V == 64) {                                          \
      constexpr static int ROPE_HEAD_DIM_NAME = 64;                       \
      return __VA_ARGS__();                                               \
    } else {                                                              \
      assert(false);                                                      \
    }                                                                     \
  }()

// forward declaration
// template <typename Dtype, int HEAD_DIM, int ROPE_HEAD_DIM, typename Params>
// void sm80_launch_mla_kernel(const Params& params, cudaStream_t stream);

// user-facing function to run the attention kernel
template <typename Dtype, typename Params>
void sm80_run_mla(Params& params, cudaStream_t stream = nullptr) {
  // normalize params that for performance optimization
  params.normalize();

  // dispatch to proper kernel instantiation based on params
  DISPATCH_HEAD_DIM_(params.head_dim, HEAD_DIM, [&] {
    DISPATCH_ROPE_HEAD_DIM_(params.rope_head_dim, ROPE_HEAD_DIM, [&] {
      sm80_launch_mla_kernel<Dtype, HEAD_DIM, ROPE_HEAD_DIM>(params, stream);
    });
  });
}

}  // namespace llm
