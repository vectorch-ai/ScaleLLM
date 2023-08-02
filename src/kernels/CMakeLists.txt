include(cc_library)

cc_library(
  NAME 
    kernels
  HDRS 
    reduction_utils.cuh
  SRCS 
    rms_norm_kernel.cu
  DEPS
    ${TORCH_LIBRARIES}
    Python::Python
)
