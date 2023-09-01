cc_library(
  NAME 
    causal_lm
  HDRS 
    causal_lm.h
  SRCS
    causal_lm.cpp 
  DEPS
    :llama
    glog::glog
    torch
)

cc_library(
  NAME 
    model_args
  HDRS 
    model_args.h
  SRCS 
  DEPS
    torch
)

cc_library(
  NAME 
    input_parameters
  HDRS 
    input_parameters.h
  SRCS 
  DEPS
    torch
)

add_subdirectory(llama)
