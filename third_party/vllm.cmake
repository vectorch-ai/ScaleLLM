include(cc_library)

cc_library(
  NAME 
    vllm.kernels
  SRCS 
    vllm/csrc/attention/attention_kernels.cu
  DEPS
    torch
    Python::Python
)
