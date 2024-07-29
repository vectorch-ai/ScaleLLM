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
    exllama.kernels
  SRCS 
    exllama/exllama_ext.cpp
    exllama/cuda_buffers.cu
    exllama/cuda_func/column_remap.cu
    exllama/cuda_func/q4_matmul.cu
    exllama/cuda_func/q4_matrix.cu
  DEPS
    torch
  LINKOPTS 
    cublas
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
)

