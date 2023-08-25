include(cc_library)

cc_library(
  NAME 
    vllm.kernels
  SRCS 
    vllm/csrc/cache.cpp
    vllm/csrc/cache_kernels.cu
    vllm/csrc/attention.cpp
    vllm/csrc/attention/attention_kernels.cu
    # vllm/csrc/pos_encoding.cpp
    # vllm/csrc/pos_encoding_kernels.cu
    # vllm/csrc/layernorm.cpp
    # vllm/csrc/layernorm_kernels.cu
    # vllm/csrc/activation.cpp
    # vllm/csrc/activation_kernels.cu
  DEPS
    torch
    Python::Python
  COPTS
    # "$<$<COMPILE_LANGUAGE:CUDA>:-G>"
)
