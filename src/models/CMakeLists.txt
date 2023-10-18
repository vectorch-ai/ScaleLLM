include(cc_library)
include(cc_test)

cc_library(
  NAME 
    models
  HDRS
    args.h
    input_parameters.h
    model_registry.h
    causal_lm.h
  SRCS
    model_registry.cpp
    causal_lm.cpp
  DEPS
    :common
    :layers
    :memory
    glog::glog
    torch
)

cc_test(
  NAME
    models_test
  SRCS
    llama_test.cpp
    huggingface/mpt_test.cpp
  DEPS
    :models
    :state_dict
    GTest::gtest_main
)

