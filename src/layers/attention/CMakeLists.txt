include(cc_library)
include(cc_test)


cc_library(
  NAME 
    attention
  HDRS 
    attention.h
    attention_rope.h
    attention_alibi.h
  SRCS 
    attention.cpp
    attention_rope.cpp
    attention_alibi.cpp
  DEPS
    :state_dict
    :memory
    :pos_embedding
    :kernels
    :flash_attn.kernels
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
    GTest::gtest_main
)
