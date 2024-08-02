include(cc_library)
include(cc_test)

cc_library(
  NAME 
    triton_example.kernel
  HDRS 
    aot/add_kernel_fp16.h
    aot/add_kernel_fp32.h
    aot/fp16/add_kernel_fp16.6a55b24f_0123.h
    aot/fp32/add_kernel_fp32.3fbfba5d_0123.h
  SRCS 
    aot/add_kernel_fp16.c
    aot/add_kernel_fp32.c
    aot/fp16/add_kernel_fp16.6a55b24f_0123.c
    aot/fp32/add_kernel_fp32.3fbfba5d_0123.c
  DEPS
    cuda
    CUDA::toolkit
)

cc_test(
  NAME 
    triton_example_test
  SRCS
    kernel_test.cpp
  DEPS 
    triton_example.kernel
    GTest::gtest_main
    torch
)

