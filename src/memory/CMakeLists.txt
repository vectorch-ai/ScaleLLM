include(cc_library)
include(cc_test)

cc_library(
  NAME 
    memory
  HDRS 
    block_allocator.h
    block_manager.h
  SRCS 
    block_manager.cpp
  DEPS
    glog::glog
    ${TORCH_LIBRARIES}
)

# cc_test(
#   NAME
#     memory_test
#   SRCS
#     block_allocator_test.cpp
#   DEPS
#     :memory
#     GTest::gtest_main
# )
