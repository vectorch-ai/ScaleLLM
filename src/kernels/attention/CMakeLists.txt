include(cc_library)
include(cc_test)

cc_library(
  NAME 
    attention.kernels
  SRCS 
    # attention.cu
  DEPS
    cutlass
)

cc_test(
  NAME
    cute_test
  SRCS
    cute_test.cpp
  DEPS
    cutlass
    GTest::gtest_main
    torch # include torch for CUDA configuration for now
)
