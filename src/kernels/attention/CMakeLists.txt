include(cc_binary)
include(cc_library)
include(cc_test)

cc_library(
  NAME 
    attention.template
  HDRS
    ptx.cuh
    fast_cast.cuh
    online_softmax.cuh
    attention_traits_sm80.h
    attention_kernel_sm80.cuh
  DEPS
    cutlass
)

cc_library(
  NAME 
    attention.kernel
  HDRS
    # attention_launcher.cuh
    # attention.h
  SRCS 
    # attention.cpp
  DEPS
    :attention.template
    glog::glog
)

cc_test(
  NAME
    attention_kernel_test
  SRCS
    attention_cpu_test.cpp
    attention_traits_test.cpp
    attention_kernel_sm80_test.cu
  DEPS
    :attention.template
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
    :attention.template
  COPTS
    -lineinfo 
)

add_subdirectory(flash_attn)
add_subdirectory(flash_infer)
add_subdirectory(tools)