include(cc_library)
include(cc_test)

cc_library(
  NAME 
    memory
  HDRS 
    block_allocator.h
    cache_manager.h 
    memory_planner.h
  SRCS 
    block_allocator.cpp
  DEPS
    glog::glog
)

cc_test(
  NAME
    memory_test
  SRCS
    block_allocator_test.cpp
  DEPS
    :memory
    GTest::gtest_main
)
