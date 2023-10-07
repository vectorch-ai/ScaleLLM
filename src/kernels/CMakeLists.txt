include(cc_library)

cc_library(
  NAME 
    kernels
  HDRS 
    reduce_kernel_utils.cuh
    activation_kernels.h
    layernorm_kernels.h
  SRCS 
    # pos_embedding_kernel.cu
    activation_kernels.cu
    layernorm_kernels.cu
  DEPS
    glog::glog
    torch
    Python::Python
  COPTS
    "$<$<COMPILE_LANGUAGE:CUDA>:-G>"
)

cc_library(
  NAME 
    gptq.kernels
  HDRS 
  SRCS 
    gptq/gptq_kernel_64.cu
  DEPS
    torch
    Python::Python
  COPTS
    # "$<$<COMPILE_LANGUAGE:CUDA>:-G>"
)

cc_library(
  NAME 
    awq.kernels
  HDRS 
  SRCS 
    awq/gemm_cuda_gen.cu
  DEPS
    torch
    Python::Python
  COPTS
    # "$<$<COMPILE_LANGUAGE:CUDA>:-G>"
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
    Python::Python
  COPTS
    # "$<$<COMPILE_LANGUAGE:CUDA>:-G>"
)

