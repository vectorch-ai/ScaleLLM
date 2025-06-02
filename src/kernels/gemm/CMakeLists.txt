include(cc_library)
include(cc_test)

cc_library(
  NAME
    gemm.kernels
  HDRS
    grouped_gemm_kernel_sm80.cuh
  DEPS
    cutlass
)


cc_test(
  NAME
    gemm_kernel_test
  SRCS
    grouped_gemm_kernel_sm80_test.cu
  DEPS
    :gemm.kernels
    absl::random_random
    GTest::gtest_main
    torch
)
