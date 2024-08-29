include(cc_library)
include(cc_test)

# find all *.h/*.c files in folder aot recursively
file(GLOB_RECURSE AOT_HEADERS aot/*.h)
file(GLOB_RECURSE AOT_SOURCES aot/*.c)

cc_library(
  NAME 
    triton_example.kernel
  HDRS 
    ${AOT_HEADERS}
  SRCS 
    ${AOT_SOURCES}
  DEPS
    CUDA::driver
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

