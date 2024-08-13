include(cc_library)

cc_library(
  NAME 
    gptq.kernels
  HDRS 
  SRCS 
    gptq/gptq_kernel.cu
  DEPS
    torch
)

cc_library(
  NAME 
    awq.kernels
  HDRS 
  SRCS 
    awq/gemm_cuda_gen.cu
  DEPS
    torch
)

cc_library(
  NAME 
    exllamav2.kernels
  SRCS 
    exllamav2/ext.cpp
    exllamav2/cuda/q_matrix.cu
    exllamav2/cuda/q_gemm.cu
  DEPS
    torch
  LINKOPTS 
    cublas
)

cc_library(
  NAME
    marlin.kernels
  HDRS
    marlin/marlin.h
  SRCS
    marlin/fp16_int4_gemm.cu
    marlin/fp8_gemm.cu
    marlin/gptq_gemm.cu
    marlin/gptq_repack.cu
    # marlin/sparse.cu
  DEPS
    torch
)

