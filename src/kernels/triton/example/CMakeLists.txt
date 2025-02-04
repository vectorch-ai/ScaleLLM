include(cc_library)
include(cc_test)

# globbing all generated files *.h/*.c in folder aot
file(GLOB AOT_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/aot/*.cuh)
file(GLOB AOT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/aot/*.cu)

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

