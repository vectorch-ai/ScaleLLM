include(cc_library)

cc_library(
  TESTONLY
  NAME
    gtest_main
  SRCS
    gtest_main.cpp
  DEPS
    GTest::gtest
    CUDA::toolkit
  LINKOPTS
    cudart
)
