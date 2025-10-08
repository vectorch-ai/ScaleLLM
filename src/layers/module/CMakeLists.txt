include(cc_library)
include(cc_test)

cc_library(
  NAME
    module
  HDRS
    module.h
    module_list.h
    module_holder.h
  SRCS
    module.cpp
  DEPS
    glog::glog
    torch
)

cc_test(
  NAME
    module_test
  SRCS
    module_test.cpp
  DEPS
    :module
    :state_dict
    :gtest_main
    torch
)
