include(cc_library)
include(cc_test)

cc_library(
  NAME 
    scheduler
  HDRS
    scheduler.h
    response_handler.h
    scheduler_config.h
    scheduler_factory.h
    scheduler_policy.h
    continuous_scheduler.h
  SRCS 
    response_handler.cpp
    scheduler_config.cpp
    scheduler_policy.cpp
    continuous_scheduler.cpp
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
#     absl::strings
#     GTest::gtest_main
# )
