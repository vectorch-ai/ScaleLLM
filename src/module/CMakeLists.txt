include(cc_library)

cc_library(
  NAME
    module
  HDRS
    module.h
    modulelist.h
    module_holder.h
  SRCS
    module.cpp
  DEPS
    glog::glog
    torch
)
