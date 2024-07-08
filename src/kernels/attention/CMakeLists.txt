include(cc_library)
include(cc_test)

cc_library(
  NAME 
    attention.kernels
  SRCS 
    # attention.cu
  INCLUDES
    include
)

cc_test(
  NAME
    cute_test
  SRCS
    cute_test.cpp
  INCLUDES
    include
  DEPS
    GTest::gtest_main
    torch # include torch for CUDA configuration for now
)
