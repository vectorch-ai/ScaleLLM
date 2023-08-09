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

cc_library(
  NAME 
    model_args
  HDRS 
    model_args.h
  SRCS 
  DEPS
    ${TORCH_LIBRARIES}
)

add_subdirectory(llama)
