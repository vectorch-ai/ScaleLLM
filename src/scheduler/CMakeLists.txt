include(cc_library)
include(cc_test)

cc_library(
  NAME 
    scheduler
  HDRS
    response_handler.h
    scheduler.h
    scheduler_policy.h
    continuous_batching_scheduler.h
    speculative_scheduler.h
  SRCS 
    response_handler.cpp
    scheduler_policy.cpp
    continuous_batching_scheduler.cpp
    speculative_scheduler.cpp
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
