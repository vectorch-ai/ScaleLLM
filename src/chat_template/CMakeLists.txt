include(cc_library)
include(cc_test)

cc_library (
  NAME
    chat_template
  HDRS
    chat_template.h
    coded_chat_template.h
    common_chat_template.h
  SRCS
    coded_chat_template.cpp
    common_chat_template.cpp
  DEPS
    glog::glog
)
