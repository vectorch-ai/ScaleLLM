include(cc_library)
include(cc_test)

cc_library(
  NAME 
    request
  HDRS 
    sampling_parameter.h 
    stopping_criteria.h
    sequence.h
    request.h
  SRCS 
    sequence.cpp
  DEPS
    glog::glog
)
