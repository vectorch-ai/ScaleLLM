cc_library(
  NAME 
    causal_lm
  HDRS 
    causal_lm.h
  SRCS 
  DEPS
    :layers
    glog::glog
    ${TORCH_LIBRARIES}
)

add_subdirectory(llama)
