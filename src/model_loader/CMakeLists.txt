include(cc_library)

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
