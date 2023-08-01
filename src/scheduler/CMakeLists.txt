include(cc_library)
include(cc_test)

cc_library(
  NAME 
    scheduler
  HDRS 
    request_context.h
    request_queue.h
    scheduler.h
    naive_batching_scheduler.h
  SRCS 
    request_queue.cpp
    naive_batching_scheduler.cpp
  DEPS
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
