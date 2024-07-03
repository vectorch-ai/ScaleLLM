include(cc_library)
include(cc_test)

cc_library(
  NAME 
    models
  HDRS
    model_args.h
    parameters.h
    model_registry.h
    causal_lm.h
  SRCS
    model_registry.cpp
    causal_lm.cpp
  DEPS
    :common
    :layers
    :quantization
    :memory
    :chat_template
    glog::glog
    torch
)

