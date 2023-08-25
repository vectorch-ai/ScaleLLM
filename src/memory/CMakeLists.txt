include(cc_library)
include(cc_test)

cc_library(
  NAME 
    memory
  HDRS 
    kv_cache.h
    block_allocator.h
    block_manager.h
  SRCS 
    kv_cache.cpp
    block_manager.cpp
  DEPS
    :vllm.kernels
    glog::glog
    ${TORCH_LIBRARIES}
)

cc_test(
  NAME
    memory_test
  SRCS
    kv_cache_test.cpp
  DEPS
    :memory
    GTest::gtest_main
)
