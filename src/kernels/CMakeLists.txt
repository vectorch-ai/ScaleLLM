include(cc_library)

cc_library(
  NAME 
    kernels
  HDRS 
    reduce_kernel_utils.cuh
  SRCS 
    rms_norm_kernel.cu
  DEPS
    ${TORCH_LIBRARIES}
    Python::Python
)
