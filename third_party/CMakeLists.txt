include(cc_library)

cc_library(
  NAME
    cutlass
  INCLUDES
    cutlass/include
  DEPS
    torch # TODO: depends on CUDA instead of torch
)

add_subdirectory(sentencepiece)
if (BUILD_NVBENCH)
  add_subdirectory(nvbench)
endif()
