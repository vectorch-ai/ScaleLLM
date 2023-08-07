include(cc_library)
include(cc_test)

cc_library(
  NAME 
    request
  HDRS 
    sequence.h
    request.h
  SRCS 
  DEPS
    glog::glog
)
