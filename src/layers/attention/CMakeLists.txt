include(cc_library)
include(cc_test)


cc_library(
  NAME 
    attention
  HDRS 
    handler.h
    ref_handler.h
    scale_attn_handler.h
    attention.h
  SRCS 
    handler.cpp
    ref_handler.cpp
    scale_attn_handler.cpp
    attention.cpp
  DEPS
    :state_dict
    :memory
    :pos_embedding
    :kernels
    :attention.kernels
    glog::glog
    gflags::gflags
    torch
)

cc_test(
  NAME
    attention_test
  SRCS
    attention_test.cpp
  DEPS
    :attention
    absl::random_random
    :gtest_main
)
