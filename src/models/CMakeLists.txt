include(cc_library)
cc_library(
  NAME 
    layers
  HDRS 
    embedding.h
    linear.h

  SRCS 
    embedding.cpp
    linear.cpp
  DEPS
    :torch_utils
    glog::glog
    ${TORCH_LIBRARIES}
)

add_subdirectory(llama)
