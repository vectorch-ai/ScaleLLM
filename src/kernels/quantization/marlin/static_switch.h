#pragma once

#include <glog/logging.h>

#define BOOL_SWITCH(cond, CONST_NAME, ...)      \
  [&] {                                         \
    if (cond) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

// FLOAT_TYPE: [Half, BFloat16]
#define FLOAT_TYPE_SWITCH(type, ...)                    \
  [&] {                                                 \
    if (type == at::ScalarType::Half) {                 \
      using scalar_t = half;                            \
      return __VA_ARGS__();                             \
    } else if (type == at::ScalarType::BFloat16) {      \
      using scalar_t = nv_bfloat16;                     \
      return __VA_ARGS__();                             \
    } else {                                            \
      LOG(FATAL) << "Unsupported FLOAT_TYPE: " << type; \
    }                                                   \
  }()

// NUM_BITS: [4, 8]
#define NUM_BITS_SWITCH(num_bits, ...)                    \
  [&] {                                                   \
    if (num_bits == 4) {                                  \
      constexpr static int NUM_BITS = 4;                  \
      return __VA_ARGS__();                               \
    } else if (num_bits == 8) {                           \
      constexpr static int NUM_BITS = 8;                  \
      return __VA_ARGS__();                               \
    } else {                                              \
      LOG(FATAL) << "Unsupported num_bits: " << num_bits; \
    }                                                     \
  }()

// M_BLOCKS: 1, 2, 3, 4
#define M_BLOCKS_SWITCH(thread_m_blocks, ...)                           \
  [&] {                                                                 \
    if (thread_m_blocks == 1) {                                         \
      constexpr static int THREAD_M_BLOCKS = 1;                         \
      return __VA_ARGS__();                                             \
    } else if (thread_m_blocks == 2) {                                  \
      constexpr static int THREAD_M_BLOCKS = 2;                         \
      return __VA_ARGS__();                                             \
    } else if (thread_m_blocks == 3) {                                  \
      constexpr static int THREAD_M_BLOCKS = 3;                         \
      return __VA_ARGS__();                                             \
    } else if (thread_m_blocks == 4) {                                  \
      constexpr static int THREAD_M_BLOCKS = 4;                         \
      return __VA_ARGS__();                                             \
    } else {                                                            \
      LOG(FATAL) << "Unsupported thread_m_blocks: " << thread_m_blocks; \
    }                                                                   \
  }()

#define DISPATCH_NK_BLOCKS_THREADS(N_BLOCKS, K_BLOCKS, THREADS, ...) \
  if (N_BLOCKS == thread_n_blocks && K_BLOCKS == thread_k_blocks &&  \
      THREADS == num_threads) {                                      \
    constexpr static int THREAD_N_BLOCKS = N_BLOCKS;                 \
    constexpr static int THREAD_K_BLOCKS = K_BLOCKS;                 \
    constexpr static int NUM_THREADS = THREADS;                      \
    return __VA_ARGS__();                                            \
  }

// HAS_ZP: [true, false]
#define HAS_ZP_SWITCH(has_zp, ...)          \
  [&] {                                     \
    if (has_zp) {                           \
      constexpr static bool HAS_ZP = true;  \
      return __VA_ARGS__();                 \
    } else {                                \
      constexpr static bool HAS_ZP = false; \
      return __VA_ARGS__();                 \
    }                                       \
  }()

// (ACT_ORDER, GROUP): (true, 0), (false, [-1, 2, 4, 8])
#define ACT_ORDER_GROUP_BLOCKS_SWITCH(act_order, group_blocks, ...) \
  [&] {                                                             \
    if (!HAS_ZP && act_order) {                                     \
      constexpr static bool HAS_ACT_ORDER = true;                   \
      if (group_blocks == 0) {                                      \
        constexpr static int GROUP_BLOCKS = 0;                      \
        return __VA_ARGS__();                                       \
      } else {                                                      \
        LOG(FATAL) << "Unsupported group_blocks: " << group_blocks; \
      }                                                             \
    } else {                                                        \
      constexpr static bool HAS_ACT_ORDER = false;                  \
      if (group_blocks == -1) {                                     \
        constexpr static int GROUP_BLOCKS = -1;                     \
        return __VA_ARGS__();                                       \
      } else if (group_blocks == 2) {                               \
        constexpr static int GROUP_BLOCKS = 2;                      \
        return __VA_ARGS__();                                       \
      } else if (group_blocks == 4) {                               \
        constexpr static int GROUP_BLOCKS = 4;                      \
        return __VA_ARGS__();                                       \
      } else if (group_blocks == 8) {                               \
        constexpr static int GROUP_BLOCKS = 8;                      \
        return __VA_ARGS__();                                       \
      } else {                                                      \
        LOG(FATAL) << "Unsupported group_blocks: " << group_blocks; \
      }                                                             \
    }                                                               \
  }()
