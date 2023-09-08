include(cc_library)
include(cc_test)

cc_library(
  NAME 
    scheduler
  HDRS
    scheduler.h
    continuous_batching_scheduler.h
  SRCS 
    continuous_batching_scheduler.cpp
  DEPS
    :request
    :engine
    glog::glog
    Folly::folly
    absl::time
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
