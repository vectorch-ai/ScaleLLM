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
  SRCS
    ${GENERATED_SRC_FILES}
    fp16_int4_gemm.cu
    fp8_gemm.cu
    gptq_gemm.cu
    gptq_repack.cu
    awq_repack.cu
    # sparse.cu
  INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}
  DEPS
    torch
)

