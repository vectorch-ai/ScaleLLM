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
    conversation.h
  SRCS
    model_registry.cpp
    causal_lm.cpp
    conversation.cpp
  DEPS
    :common
    :layers
    :memory
    glog::glog
    torch
)

