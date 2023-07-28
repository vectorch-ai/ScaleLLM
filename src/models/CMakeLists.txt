include(cc_library)
include(cc_test)
cc_library(
  NAME 
    layers
  HDRS 
    layers.h
  SRCS 
    layers.cpp
  DEPS
    :torch_utils
    glog::glog
    ${TORCH_LIBRARIES}
)

cc_test(
  NAME
    layers_test
  SRCS
    layers_test.cpp
  DEPS
    :layers
    :torch_utils
    GTest::gtest_main
)

add_subdirectory(llama)
