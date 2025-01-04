include(cc_binary)
include(cc_library)
include(cc_test)

cc_library(
  NAME 
    attention.kernel
  HDRS
    attention_cpu.h
    attention_kernel_sm80.cuh
  SRCS 
    # attention.cu
  DEPS
    cutlass
    glog::glog
)

cc_test(
  NAME
    attention_kernel_test
  SRCS
    cute_test.cpp
    attention_cpu_test.cpp
    attention_kernel_sm80_test.cu
  DEPS
    :attention.kernel
    glog::glog
    GTest::gtest_main
)

cc_binary(
  NAME 
    attention_sm80_bench
  SRCS 
    attention_sm80_bench.cu
  DEPS
    nvbench::nvbench
    nvbench::main
    :attention.kernel
)



add_subdirectory(flash_attn)
add_subdirectory(flash_infer)