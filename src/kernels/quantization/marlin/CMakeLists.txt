include(cc_library)

# copy the file to binary dir in order to force re-configuration when the file changes
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/generate_instantiations.py
  ${CMAKE_CURRENT_BINARY_DIR}/generate_instantiations.py
)

# execute the script to generate the instantiation of the kernels
execute_process(
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/generate_instantiations.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/
)

# globbing all generated files in sub directory "generated"
file(GLOB GENERATED_SRC_FILES "${CMAKE_CURRENT_BINARY_DIR}/generated/*.cu")

cc_library(
  NAME
    marlin.kernels
  HDRS
    marlin.h
    gemm_kernel.cuh
  SRCS
    fp16_int4_gemm.cu
    fp8_gemm.cu
    gptq_gemm.cu
    gptq_repack.cu
    awq_repack.cu
    # sparse.cu
    ${GENERATED_SRC_FILES}
    # marlin_fp16_b4_t256_m4_n16_k4_s4_false_false_g8.cu
    # marlin_bf16_b4_t256_m4_n16_k4_s4_false_false_g8.cu
  INCLUDES
    .
  DEPS
    torch
)

