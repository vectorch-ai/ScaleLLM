#pragma once

namespace marlin {
template <typename scalar_t,          // compute dtype, half or nv_float16
          const int num_bits,         // number of bits used for weights
          const int threads,          // number of threads in a threadblock
          const int thread_m_blocks,  // number of 16x16 blocks in the m
                                      // dimension (batchsize) of the
                                      // threadblock
          const int thread_n_blocks,  // same for n dimension (output)
          const int thread_k_blocks,  // same for k dimension (reduction)
          const int stages,  // number of stages for the async global->shared
                             // fetch pipeline
          const bool has_act_order,    // whether act_order is enabled
          const bool has_zp,           // whether zero-points are enabled
          const int group_blocks       // number of consecutive 16x16 blocks
                                       // with a separate quantization scale
          >
void Marlin(
    const int4* __restrict__ A,  // fp16 input matrix of shape mxk
    const int4* __restrict__ B,  // 4bit quantized weight matrix of shape kxn
    int4* __restrict__ C,        // fp16 output buffer of shape mxn
    int4* __restrict__ C_tmp,    // fp32 tmp output buffer (for reduce)
    const int4* __restrict__ scales_ptr,  // fp16 quantization scales of shape
                                          // (k/groupsize)xn
    const int4* __restrict__ zp_ptr,      // 4bit packed zero-points of shape
                                          // (k/groupsize)x(n/pack_factor)
    const int* __restrict__ g_idx,        // int32 group indices of shape k
    int num_groups,       // number of scale groups per output channel
    int prob_m,           // batch dimension m
    int prob_n,           // output dimension n
    int prob_k,           // reduction dimension k
    int* locks,           // extra global storage for barrier synchronization
    bool use_fp32_reduce  // whether to use fp32 global reduce
);

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

#define GPTQ_CALL_IF(NUM_BITS, N_BLOCKS, K_BLOCKS, NUM_THREADS)             \
  __CALL_IF(NUM_BITS, 1, N_BLOCKS, K_BLOCKS, true, false, 0, NUM_THREADS)   \
  __CALL_IF(NUM_BITS, 2, N_BLOCKS, K_BLOCKS, true, false, 0, NUM_THREADS)   \
  __CALL_IF(NUM_BITS, 3, N_BLOCKS, K_BLOCKS, true, false, 0, NUM_THREADS)   \
  __CALL_IF(NUM_BITS, 4, N_BLOCKS, K_BLOCKS, true, false, 0, NUM_THREADS)   \
                                                                            \
  __CALL_IF(NUM_BITS, 1, N_BLOCKS, K_BLOCKS, false, false, -1, NUM_THREADS) \
  __CALL_IF(NUM_BITS, 1, N_BLOCKS, K_BLOCKS, false, false, 2, NUM_THREADS)  \
  __CALL_IF(NUM_BITS, 1, N_BLOCKS, K_BLOCKS, false, false, 4, NUM_THREADS)  \
  __CALL_IF(NUM_BITS, 1, N_BLOCKS, K_BLOCKS, false, false, 8, NUM_THREADS)  \
                                                                            \
  __CALL_IF(NUM_BITS, 2, N_BLOCKS, K_BLOCKS, false, false, -1, NUM_THREADS) \
  __CALL_IF(NUM_BITS, 2, N_BLOCKS, K_BLOCKS, false, false, 2, NUM_THREADS)  \
  __CALL_IF(NUM_BITS, 2, N_BLOCKS, K_BLOCKS, false, false, 4, NUM_THREADS)  \
  __CALL_IF(NUM_BITS, 2, N_BLOCKS, K_BLOCKS, false, false, 8, NUM_THREADS)  \
                                                                            \
  __CALL_IF(NUM_BITS, 3, N_BLOCKS, K_BLOCKS, false, false, -1, NUM_THREADS) \
  __CALL_IF(NUM_BITS, 3, N_BLOCKS, K_BLOCKS, false, false, 2, NUM_THREADS)  \
  __CALL_IF(NUM_BITS, 3, N_BLOCKS, K_BLOCKS, false, false, 4, NUM_THREADS)  \
  __CALL_IF(NUM_BITS, 3, N_BLOCKS, K_BLOCKS, false, false, 8, NUM_THREADS)  \
                                                                            \
  __CALL_IF(NUM_BITS, 4, N_BLOCKS, K_BLOCKS, false, false, -1, NUM_THREADS) \
  __CALL_IF(NUM_BITS, 4, N_BLOCKS, K_BLOCKS, false, false, 2, NUM_THREADS)  \
  __CALL_IF(NUM_BITS, 4, N_BLOCKS, K_BLOCKS, false, false, 4, NUM_THREADS)  \
  __CALL_IF(NUM_BITS, 4, N_BLOCKS, K_BLOCKS, false, false, 8, NUM_THREADS)

#define AWQ_CALL_IF(NUM_BITS, N_BLOCKS, K_BLOCKS, NUM_THREADS)             \
  __CALL_IF(NUM_BITS, 1, N_BLOCKS, K_BLOCKS, false, true, -1, NUM_THREADS) \
  __CALL_IF(NUM_BITS, 1, N_BLOCKS, K_BLOCKS, false, true, 2, NUM_THREADS)  \
  __CALL_IF(NUM_BITS, 1, N_BLOCKS, K_BLOCKS, false, true, 4, NUM_THREADS)  \
  __CALL_IF(NUM_BITS, 1, N_BLOCKS, K_BLOCKS, false, true, 8, NUM_THREADS)  \
                                                                           \
  __CALL_IF(NUM_BITS, 2, N_BLOCKS, K_BLOCKS, false, true, -1, NUM_THREADS) \
  __CALL_IF(NUM_BITS, 2, N_BLOCKS, K_BLOCKS, false, true, 2, NUM_THREADS)  \
  __CALL_IF(NUM_BITS, 2, N_BLOCKS, K_BLOCKS, false, true, 4, NUM_THREADS)  \
  __CALL_IF(NUM_BITS, 2, N_BLOCKS, K_BLOCKS, false, true, 8, NUM_THREADS)  \
                                                                           \
  __CALL_IF(NUM_BITS, 3, N_BLOCKS, K_BLOCKS, false, true, -1, NUM_THREADS) \
  __CALL_IF(NUM_BITS, 3, N_BLOCKS, K_BLOCKS, false, true, 2, NUM_THREADS)  \
  __CALL_IF(NUM_BITS, 3, N_BLOCKS, K_BLOCKS, false, true, 4, NUM_THREADS)  \
  __CALL_IF(NUM_BITS, 3, N_BLOCKS, K_BLOCKS, false, true, 8, NUM_THREADS)  \
                                                                           \
  __CALL_IF(NUM_BITS, 4, N_BLOCKS, K_BLOCKS, false, true, -1, NUM_THREADS) \
  __CALL_IF(NUM_BITS, 4, N_BLOCKS, K_BLOCKS, false, true, 2, NUM_THREADS)  \
  __CALL_IF(NUM_BITS, 4, N_BLOCKS, K_BLOCKS, false, true, 4, NUM_THREADS)  \
  __CALL_IF(NUM_BITS, 4, N_BLOCKS, K_BLOCKS, false, true, 8, NUM_THREADS)

}  // namespace marlin