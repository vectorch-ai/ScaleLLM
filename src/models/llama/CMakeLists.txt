include(cc_library)

cc_library(
  NAME 
    llama
  HDRS 
    attention.h
    feedforward.h
    transformer_block.h
    transformer.h
  SRCS 
    attention.cpp
    transformer.cpp
  DEPS
    :layers
    glog::glog
    ${TORCH_LIBRARIES}
)
