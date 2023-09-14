include(cc_library)

cc_library(
  NAME 
    state_dict
  HDRS 
    state_dict.h
  SRCS 
    state_dict.cpp
  DEPS
    glog::glog
    torch
)

cc_library(
  NAME 
    model_loader
  HDRS 
    model_loader.h
  SRCS 
    model_loader.cpp
  DEPS
    :model_args
    torch
)

add_subdirectory(safetensors)
