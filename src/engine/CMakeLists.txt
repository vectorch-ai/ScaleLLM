include(cc_library)
include(cc_test)

cc_library(
  NAME 
    engine
  HDRS
    engine.h
    worker.h
  SRCS 
  DEPS
    :request
    glog::glog
    Folly::folly
)

# cc_test(
#   NAME
#     scheduler_test
#   SRCS
#     scheduler_test.cpp
#   DEPS
#     :scheduler
#     GTest::gtest_main
# )
