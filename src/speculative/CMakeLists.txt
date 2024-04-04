include(cc_library)
include(cc_test)

cc_library(
  NAME 
    speculative
  HDRS
    rejection_sampler.h
    speculative_engine.h
  SRCS 
    rejection_sampler.cpp
    speculative_engine.cpp
  DEPS
    :engine
    :sampler
    glog::glog
    Folly::folly
    absl::time
)

cc_test(
  NAME
    speculative_test
  SRCS
    # speculative_test.cpp
    rejection_sampler_test.cpp
  DEPS
    :speculative
    absl::strings
    GTest::gtest_main
)
