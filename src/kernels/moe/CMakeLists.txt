include(cc_library)
include(cc_test)

cc_library(
  NAME
    moe.kernels
  SRCS
    topk_softmax_kernel.cu
    grouped_topk_sigmoid_kernel.cu
    permutation_index_kernel.cu
    permutation_mask_kernel.cu
    align_block_kernel.cu
  DEPS
    cutlass
    glog::glog
    torch
)


cc_test(
  NAME
    moe_kernel_test
  SRCS
    # topk_softmax_kernel_test.cu
    # grouped_topk_sigmoid_kernel_test.cu
    # permutation_kernel_test.cu
    align_block_kernel_test.cu
  DEPS
    :moe.kernels
    absl::random_random
    GTest::gtest_main
)
