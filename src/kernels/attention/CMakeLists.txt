include(cc_library)
include(cc_test)

cc_library(
  NAME 
    attention.kernel
  SRCS 
    # attention.cu
  DEPS
    cutlass
    glog::glog
)

cc_test(
  NAME
    attention_kernel_test
  HDRS
    attention_cpu.h
  SRCS
    cute_test.cpp
    attention_test.cpp
  DEPS
    :attention.kernel
    glog::glog
    GTest::gtest_main
)
