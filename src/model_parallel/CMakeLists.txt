include(cc_library)
include(cc_test)

cc_library(
  NAME 
    process_group
  HDRS
    process_group.h
  SRCS
    process_group.cpp
  DEPS
    torch
    NCCL::nccl
    glog::glog
)

cc_library(
  NAME 
    model_parallel
  HDRS
    parallel_args.h
    model_parallel.h
  SRCS
    model_parallel.cpp
  DEPS
    :process_group
)

cc_test(
  NAME
    process_group_test
  SRCS
    process_group_test.cpp
  DEPS
    :process_group
    :gtest_main
)

