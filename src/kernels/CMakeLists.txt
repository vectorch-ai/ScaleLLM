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


# copy the file to binary dir in order to force re-configuration when the file changes
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/flash_attn/generate_instantiation_cu.py
  ${CMAKE_CURRENT_BINARY_DIR}/flash_attn/generate_instantiation_cu.py
)

# execute the script to generate the instantiation of the kernels
execute_process(
  COMMAND ${CMAKE_CURRENT_BINARY_DIR}/flash_attn/generate_instantiation_cu.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/flash_attn/
)

# globbing all generated files in sub directory "generated"
file(GLOB GENERATED_SRC_FILES "${CMAKE_CURRENT_BINARY_DIR}/flash_attn/generated/*.cu")

cc_library(
  NAME 
    flash-attn.kernels
  HDRS
    flash_attn/flash_api.h
  SRCS
    flash_attn/flash_api.cpp
    ${GENERATED_SRC_FILES}
  INCLUDES
    flash_attn
    flash_attn/src
    cutlass/include
  DEPS
    torch
)

add_subdirectory(flashinfer)

