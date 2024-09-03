include(cc_library)

# copy the file to binary dir in order to force re-configuration when the file changes
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/generate_instantiation_cu.py
  ${CMAKE_CURRENT_BINARY_DIR}/generate_instantiation_cu.py
)

# execute the script to generate the instantiation of the kernels
execute_process(
  COMMAND "${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_instantiation_cu.py"
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# globbing all generated files in sub directory "generated"
file(GLOB GENERATED_SRC_FILES "${CMAKE_CURRENT_BINARY_DIR}/generated/*.cu")

cc_library(
  NAME 
    flash_infer.kernels
  HDRS
    flashinfer_ops.h
  SRCS
    single_decode.cu
    single_prefill.cu
    batch_decode.cu
    batch_prefill.cu
    cascade.cu
    page.cu
    ${GENERATED_SRC_FILES}
  INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
  DEFINES
    FLASHINFER_ENABLE_BF16
    # FLASHINFER_ENABLE_FP8
  DEPS
    flashinfer
    torch
)


