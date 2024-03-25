include(cc_library)
include(cc_test)

cc_library(
  NAME 
    speculative
  HDRS
    speculative_engine.h
  SRCS 
    speculative_engine.cpp
  DEPS
    :engine
    glog::glog
    Folly::folly
    absl::time
)

# cc_test(
#   NAME
#     speculative_test
#   SRCS
#     speculative_test.cpp
#   DEPS
#     :speculative
#     absl::strings
#     GTest::gtest_main
# )
