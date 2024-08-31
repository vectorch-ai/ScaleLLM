#pragma once

#include <glog/logging.h>

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

// FLOAT_TYPE: [Half, BFloat16]
#define FLOAT_TYPE_SWITCH(TYPE, ...)                    \
  [&] {                                                 \
    if (TYPE == at::ScalarType::Half) {                 \
      using scalar_t = half;                            \
      return __VA_ARGS__();                             \
    } else if (TYPE == at::ScalarType::BFloat16) {      \
      using scalar_t = nv_bfloat16;                     \
      return __VA_ARGS__();                             \
    } else {                                            \
      LOG(FATAL) << "Unsupported FLOAT_TYPE: " << TYPE; \
    }                                                   \
  }()

// NUM_BITS: [4, 8]
#define NUM_BITS_SWITCH(BITS, ...)                    \
  [&] {                                               \
    if (BITS == 4) {                                  \
      constexpr static int NUM_BITS = 4;              \
      return __VA_ARGS__();                           \
    } else if (BITS == 8) {                           \
      constexpr static int NUM_BITS = 8;              \
      return __VA_ARGS__();                           \
    } else {                                          \
      LOG(FATAL) << "Unsupported NUM_BITS: " << BITS; \
    }                                                 \
  }()

// M_BLOCKS: 1, 2, 3, 4
#define M_BLOCKS_SWITCH(BLOCKS, ...)                      \
  [&] {                                                   \
    if (BLOCKS == 1) {                                    \
      constexpr static int THREAD_M_BLOCKS = 1;           \
      return __VA_ARGS__();                               \
    } else if (M_BLOCKS == 2) {                           \
      constexpr static int THREAD_M_BLOCKS = 2;           \
      return __VA_ARGS__();                               \
    } else if (M_BLOCKS == 3) {                           \
      constexpr static int THREAD_M_BLOCKS = 3;           \
      return __VA_ARGS__();                               \
    } else if (M_BLOCKS == 4) {                           \
      constexpr static int THREAD_M_BLOCKS = 4;           \
      return __VA_ARGS__();                               \
    } else {                                              \
      LOG(FATAL) << "Unsupported M_BLOCKS: " << M_BLOCKS; \
    }                                                     \
  }()

#define DISPATCH_NK_BLOCKS(n_blocks, k_blocks, n_threads, ...) \
  if (N_BLOCKS == n_blocks && K_BLOCKS == k_blocks &&          \
      NUM_THREADS == n_threads) {                              \
    constexpr static int THREAD_N_BLOCKS = n_blocks;           \
    constexpr static int THREAD_K_BLOCKS = k_blocks;           \
    constexpr static int THREADS = n_threads;                  \
    return __VA_ARGS__();                                      \
  }

// (N_BLOCKS, K_BLOCKS, NUM_THREADS) :
//  (16, 4, 256), (8, 8, 256), (8, 4, 128), (4, 8, 128)
#define NK_BLOCKS_SWITCH(N_BLOCKS, K_BLOCKS, NUM_THREADS, ...)         \
  [&] {                                                                \
    DISPATCH_NK_BLOCKS(16, 4, 256, __VA_ARGS__);                       \
    DISPATCH_NK_BLOCKS(8, 8, 256, __VA_ARGS__);                        \
    DISPATCH_NK_BLOCKS(8, 4, 128, __VA_ARGS__);                        \
    DISPATCH_NK_BLOCKS(4, 8, 128, __VA_ARGS__);                        \
    LOG(FATAL) << "Unsupported (N_BLOCKS, K_BLOCKS, NUM_THREADS): "    \
               << N_BLOCKS << ", " << K_BLOCKS << ", " << NUM_THREADS; \
  }()

// HAS_ZP: [true, false]
#define HAS_ZP_SWITCH(HAS_ZP, ...)          \
  [&] {                                     \
    if (HAS_ZP) {                           \
      constexpr static bool HAS_ZP = true;  \
      return __VA_ARGS__();                 \
    } else {                                \
      constexpr static bool HAS_ZP = false; \
      return __VA_ARGS__();                 \
    }                                       \
  }()

// (ACT_ORDER, GROUP): (true, 0), (false, [-1, 2, 4, 8])
#define ACT_ORDER_SWITCH(ACT_ORDER, GROUP, ...)       \
  [&] {                                               \
    if (ACT_ORDER) {                                  \
      constexpr static bool HAS_ACT_ORDER = true;     \
      if (GROUP == 0) {                               \
        constexpr static int GROUP_BLOCKS = 0;        \
        return __VA_ARGS__();                         \
      } else {                                        \
        LOG(FATAL) << "Unsupported GROUP: " << GROUP; \
      }                                               \
    } else {                                          \
      constexpr static bool HAS_ACT_ORDER = false;    \
      if (GROUP == -1) {                              \
        constexpr static int GROUP_BLOCKS = -1;       \
        return __VA_ARGS__();                         \
      } else if (GROUP == 2) {                        \
        constexpr static int GROUP_BLOCKS = 2;        \
        return __VA_ARGS__();                         \
      } else if (GROUP == 4) {                        \
        constexpr static int GROUP_BLOCKS = 4;        \
        return __VA_ARGS__();                         \
      } else if (GROUP == 8) {                        \
        constexpr static int GROUP_BLOCKS = 8;        \
        return __VA_ARGS__();                         \
      } else {                                        \
        LOG(FATAL) << "Unsupported GROUP: " << GROUP; \
      }                                               \
    }                                                 \
  }()

// GPTQ
// 2: NUM_BITS: [4, 8]
// 4: (N_BLOCKS, K_BLOCKS, NUM_THREADS) : (16, 4, 256), (8, 8, 256), (8, 4,
// 128), (4, 8, 128)

// 4: M_BLOCKS: 1, 2, 3, 4
// 1: HAS_ZP: false

// 5 : (ACT_ORDER, GROUP): (true, 0), (false, [-1, 2, 4, 8])

// AWQ
// 2: NUM_BITS: [4, 8]
// 4: (N_BLOCKS, K_BLOCKS, NUM_THREADS) : (16, 4, 256), (8, 8, 256), (8, 4,
// 128), (4, 8, 128)

// 4: M_BLOCKS: 1, 2, 3, 4
// 1: HAS_ZP: true

// 4 : (ACT_ORDER, GROUP): (false, [-1, 2, 4, 8])
