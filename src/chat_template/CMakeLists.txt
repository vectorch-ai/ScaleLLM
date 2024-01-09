include(cc_library)
include(cc_test)

cc_library (
  NAME
    chat_template
  HDRS
    chat_template.h
    jinja_chat_template.h
  SRCS
    jinja_chat_template.cpp
  DEPS
    :jinja2cpp
    nlohmann_json::nlohmann_json
    grpc_proto::completion
    glog::glog
)

cc_test (
  NAME
    chat_template_test
  SRCS
    jinja_chat_template_test.cpp
  DEPS
    :chat_template
    GTest::gtest_main
)

