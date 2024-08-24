// Splitting the different head dimensions to different files to speed up
// compilation. This file is auto-generated. See "generate_instantiations.py"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "gemm_kernel.cuh"

namespace marlin {

template <>
void Marlin<half,
            /*num_bits=*/4,
            /*threads=*/256,
            /*thread_m_blocks=*/4,
            /*thread_n_blocks=*/16,
            /*thread_k_blocks=*/4,
            /*stages=*/4,
            /*has_act_order=*/false,
            /*has_zp=*/false,
            /*group_blocks=*/8>(
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

}  // namespace marlin