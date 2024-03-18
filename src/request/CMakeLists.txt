include(cc_library)
include(cc_test)

cc_library(
  NAME 
    request
  HDRS 
    sampling_parameters.h 
    stopping_criteria.h
    sequence.h
    status.h
    request.h
  SRCS 
    sequence.cpp
    request.cpp
  DEPS
    :memory
    glog::glog
    absl::strings
    absl::time
    torch
)


cc_test(
  NAME
    request_test
  SRCS
    sequence_test.cpp
  DEPS
    :request
    GTest::gtest_main
)