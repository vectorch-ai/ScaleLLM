include(cc_library)
include(cc_test)

cc_library(
  NAME 
    scheduler
  HDRS
    scheduler.h
    response_handler.h
    continuous_scheduler.h
  SRCS 
    response_handler.cpp
    continuous_scheduler.cpp
  DEPS
    :request
    :engine
    :speculative
    glog::glog
    Folly::folly
    absl::time
    absl::synchronization
)

# cc_test(
#   NAME
#     scheduler_test
#   SRCS
#     scheduler_test.cpp
#   DEPS
#     :scheduler
#     absl::strings
#     :gtest_main
# )
