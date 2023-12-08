include(cc_library)
include(cc_test)

cc_library(
  NAME 
    logits_processor
  HDRS 
    logits_processor.h
  SRCS 
    logits_processor.cpp
  DEPS
    :kernels
    glog::glog
    torch
)

cc_library(
  NAME 
    sampler
  HDRS 
    sampler.h
  SRCS 
    sampler.cpp
  DEPS
    glog::glog
    torch
)

cc_test(
  NAME
    logits_processor_test
  SRCS
    logits_processor_test.cpp
  DEPS
    :logits_processor
    GTest::gtest_main
)

cc_test(
  NAME
    sampler_test
  SRCS
    sampler_test.cpp
  DEPS
    :sampler
    GTest::gtest_main
)