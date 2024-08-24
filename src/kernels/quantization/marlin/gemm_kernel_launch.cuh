#pragma once

#include "gemm_kernel.cuh"

namespace marlin {
#define __CALL_IF(NUM_BITS,                                                   \
                  THREAD_M_BLOCKS,                                            \
                  THREAD_N_BLOCKS,                                            \
                  THREAD_K_BLOCKS,                                            \
                  HAS_ACT_ORDER,                                              \
                  HAS_ZP,                                                     \
                  GROUP_BLOCKS,                                               \
                  NUM_THREADS)                                                \
  else if (num_bits == NUM_BITS && thread_m_blocks == THREAD_M_BLOCKS &&      \
           thread_n_blocks == THREAD_N_BLOCKS &&                              \
           thread_k_blocks == THREAD_K_BLOCKS &&                              \
           has_act_order == HAS_ACT_ORDER && has_zp == HAS_ZP &&              \
           group_blocks == GROUP_BLOCKS && num_threads == NUM_THREADS) {      \
    auto kernel = &Marlin<scalar_t,                                           \
                          NUM_BITS,                                           \
                          NUM_THREADS,                                        \
                          THREAD_M_BLOCKS,                                    \
                          THREAD_N_BLOCKS,                                    \
                          THREAD_K_BLOCKS,                                    \
                          pipe_stages,                                        \
                          HAS_ACT_ORDER,                                      \
                          HAS_ZP,                                             \
                          GROUP_BLOCKS>;                                      \
    cudaFuncSetAttribute(                                                     \
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem); \
                                                                              \
    kernel<<<blocks, NUM_THREADS, max_shared_mem, stream>>>(A_ptr,            \
                                                            B_ptr,            \
                                                            C_ptr,            \
                                                            C_tmp_ptr,        \
                                                            s_ptr,            \
                                                            zp_ptr,           \
                                                            g_idx_ptr,        \
                                                            num_groups,       \
                                                            prob_m,           \
                                                            prob_n,           \
                                                            prob_k,           \
                                                            locks,            \
                                                            use_fp32_reduce); \
  }

#define GPTQ_CALL_IF(NUM_BITS, N_BLOCKS, K_BLOCKS, NUM_THREADS) \
  __CALL_IF(NUM_BITS, 4, N_BLOCKS, K_BLOCKS, false, false, 8, NUM_THREADS)

#define AWQ_CALL_IF(NUM_BITS, N_BLOCKS, K_BLOCKS, NUM_THREADS) \
  __CALL_IF(NUM_BITS, 4, N_BLOCKS, K_BLOCKS, false, true, 8, NUM_THREADS)

}  // namespace marlin