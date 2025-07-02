include(cc_library)
include(cc_test)

cc_library(
  NAME 
    request
  HDRS 
    stopping_criteria.h
    incremental_decoder.h
    sequence.h
    status.h
    request.h
  SRCS 
    stopping_criteria.cpp
    incremental_decoder.cpp
    sequence.cpp
    request.cpp
  DEPS
    :memory
    :tokenizer
    glog::glog
    absl::strings
    absl::time
    torch
)


cc_test(
  NAME
    request_test
  SRCS
    stopping_criteria_test.cpp
    sequence_test.cpp
  DEPS
    :request
    :gtest_main
)