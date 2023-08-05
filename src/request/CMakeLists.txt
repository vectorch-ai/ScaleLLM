include(cc_library)
include(cc_test)

cc_library(
  NAME 
    request
  HDRS 
    request_context.h
  SRCS 
  DEPS
    glog::glog
)
