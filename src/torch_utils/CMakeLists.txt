include(cc_library)

cc_library(
  NAME 
    torch_utils
  HDRS 
    state_dict.h
  SRCS 
    state_dict.cpp
  DEPS
    glog::glog
    torch
)
