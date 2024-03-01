include(cc_library)

# TODO: add a custom command to generate those files
file(GLOB_RECURSE GENERATED_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "generated/*.cu")

cc_library(
  NAME 
    flashinfer.kernels
  SRCS
    single_decode.cu
    single_prefill.cu
    batch_decode.cu
    batch_prefill.cu
    cascade.cu
    page.cu
    ${GENERATED_SRC_FILES}    
  INCLUDES
    include
  DEFINES
    FLASHINFER_ENABLE_BF16
  DEPS
    torch
)


