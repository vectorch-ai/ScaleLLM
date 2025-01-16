#pragma once

namespace llm {

#define DISPATCH_BOOL(BOOL_V, BOOL_NAME, ...)  \
  [&] {                                        \
    if (BOOL_V) {                              \
      constexpr static bool BOOL_NAME = true;  \
      return __VA_ARGS__();                    \
    } else {                                   \
      constexpr static bool BOOL_NAME = false; \
      return __VA_ARGS__();                    \
    }                                          \
  }()

#define DISPATCH_TORCH_DTYPE(TORCH_DTYPE, TYPE_NAME, ...) \
  [&] {                                                   \
    if (TORCH_DTYPE == torch::kHalf) {                    \
      using TYPE_NAME = cute::half_t;                     \
      return __VA_ARGS__();                               \
    } else if (TORCH_DTYPE == torch::kBFloat16) {         \
      using TYPE_NAME = cute::bfloat16_t;                 \
      return __VA_ARGS__();                               \
    } else {                                              \
      assert(false);                                      \
    }                                                     \
  }()

// TODO: fix build error for head_dim == 96
#define DISPATCH_HEAD_DIM(HEAD_DIM_V, HEAD_DIM_NAME, ...) \
  [&] {                                                   \
    if (HEAD_DIM_V <= 64) {                               \
      constexpr static int HEAD_DIM_NAME = 64;            \
      return __VA_ARGS__();                               \
    } else if (HEAD_DIM_V <= 128) {                       \
      constexpr static int HEAD_DIM_NAME = 128;           \
      return __VA_ARGS__();                               \
    } else if (HEAD_DIM_V <= 256) {                       \
      constexpr static int HEAD_DIM_NAME = 256;           \
      return __VA_ARGS__();                               \
    } else {                                              \
      assert(false);                                      \
    }                                                     \
  }()

}  // namespace llm