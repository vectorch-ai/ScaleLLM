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

# cc_library (
#   NAME
#     jinja_chat_template
#   HDRS
#     jinja_chat_template.h
#   SRCS
#     jinja_chat_template.cpp
#   DEPS
#     :jinja2cpp
#     nlohmann_json::nlohmann_json
#     glog::glog
# )

# cc_test (
#   NAME
#     chat_template_test
#   SRCS
#     jinja_chat_template_test.cpp
#   DEPS
#     :chat_template
#     :jinja_chat_template
#     GTest::gtest_main
# )

