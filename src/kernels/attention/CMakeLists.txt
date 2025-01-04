include(cc_binary)
include(cc_library)
include(cc_test)

cc_library(
  NAME 
    attention.kernel
  HDRS
    attention_cpu.h
    ptx.cuh 
    fast_cast.cuh
    online_softmax.cuh
    attention_traits_sm80.h
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
    attention_bench_sm80
  SRCS 
    attention_bench_sm80.cu
  DEPS
    nvbench::nvbench
    nvbench::main
    :attention.kernel
  COPTS
    -lineinfo 
)

add_subdirectory(flash_attn)
add_subdirectory(flash_infer)