include(cc_library)

cc_library(
  NAME 
    exllama.kernel_ops
  SRCS 
    exllama/exllama_ext/exllama_ext.cpp
    exllama/exllama_ext/cuda_buffers.cu
    exllama/exllama_ext/cuda_func/column_remap.cu
    exllama/exllama_ext/cuda_func/q4_matmul.cu
    exllama/exllama_ext/cuda_func/q4_matrix.cu
  DEPS
    ${TORCH_LIBRARIES}
    Python::Python
  COPTS
    # "$<$<COMPILE_LANGUAGE:CUDA>:-G>"
)
