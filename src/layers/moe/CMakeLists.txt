include(cc_library)
include(cc_test)

cc_library(
  NAME
    moe 
  HDRS
    token_dispatcher.h
    local_token_dispatcher.h
  SRCS
    local_token_dispatcher.cpp
  DEPS
    :model_parallel
    glog::glog
    gflags::gflags
    torch
)

# cc_test(
#   NAME
#     moe_test
#   SRCS
#     token_dispatcher_test.cpp
#   DEPS
#     :moe
#     absl::random_random
#     GTest::gtest_main
# )