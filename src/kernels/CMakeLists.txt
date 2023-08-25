include(cc_library)

cc_library(
  NAME 
    kernels
  HDRS 
    reduce_kernel_utils.cuh
  SRCS 
    rms_norm_kernel.cu
    pos_embedding_kernel.cu
  DEPS
    torch
    Python::Python
  COPTS
    # "$<$<COMPILE_LANGUAGE:CUDA>:-G>"
)
