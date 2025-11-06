include(cc_library)

cc_library(
  NAME
    cutlass
  INCLUDES
    cutlass/include
  DEPS
    CUDA::toolkit
  LINKOPTS
    cudart
)

if (BUILD_NVBENCH)
  add_subdirectory(nvbench)
endif()

if (NOT BUILD_KERNEL_ONLY)
  add_subdirectory(sentencepiece)
endif()
