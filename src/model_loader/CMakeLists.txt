include(cc_test)
include(cc_library)

cc_library(
  NAME 
    state_dict
  HDRS 
    state_dict.h
  SRCS 
    state_dict.cpp
  DEPS
    safetensors
    torch
    glog::glog
    Folly::folly
)

cc_test(
  NAME
    state_dict_test
  SRCS
    state_dict_test.cpp
  DEPS
    :state_dict
    GTest::gtest_main
  DATA 
    data/test.pth 
    data/test.safetensors
)

cc_library(
  NAME 
    model_loader
  HDRS 
    model_loader.h
  SRCS 
    model_loader.cpp
  DEPS
    :common
    :models
    :tokenizer
    torch
)

cc_library(
  NAME 
    model_downloader
  HDRS 
    model_downloader.h
  SRCS 
    model_downloader.cpp
  DEPS
    torch
    Python::Python
)

add_subdirectory(safetensors)
