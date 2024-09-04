include(cc_library)

cc_library(
  NAME 
    kernels
  HDRS 
    reduce_kernel_utils.cuh
    activation_kernels.h
    layernorm_kernels.h
    pos_embedding_kernels.h
    kv_cache_kernels.h
    sampling/sampling_kernels.h
  SRCS 
    activation_kernels.cu
    layernorm_kernels.cu
    pos_embedding_kernels.cu
    kv_cache_kernels.cu
    sampling/penalty_kernels.cu
    sampling/softmax_kernels.cu
    sampling/topk_kernels.cu
    sampling/topp_kernels.cu
  DEPS
    glog::glog
    torch
  DEFINES
    __CUDA_NO_HALF_OPERATORS__
)

add_subdirectory(flash_attn)
add_subdirectory(flash_infer)
add_subdirectory(attention)
add_subdirectory(quantization)
add_subdirectory(bench)
add_subdirectory(playground)
add_subdirectory(triton)

