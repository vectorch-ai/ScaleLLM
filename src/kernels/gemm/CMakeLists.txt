include(cc_library)
include(cc_test)

cc_library(
  NAME
    gemm.kernels
  HDRS
    sm80_collective_grouped_gemm.cuh
    sm80_collective_epilogue.cuh
    sm80_grouped_gemm_launch.cuh
    tile_scheduler.cuh
  DEPS
    cutlass
)


cc_test(
  NAME
    gemm_kernel_test
  SRCS
    tile_scheduler_test.cu
    sm80_grouped_gemm_test.cu
  DEPS
    :gemm.kernels
    absl::random_random
    GTest::gtest_main
    torch
)
