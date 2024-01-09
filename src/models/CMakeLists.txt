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
    :process_group
    :layers
    :memory
    :chat_template
    glog::glog
    torch
)

