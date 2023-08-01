include(cc_binary)

cc_binary(
  NAME 
    simple
  SRCS 
    simple.cpp
  DEPS
    :torch_utils
    :llama
    :tokenizer
    ${TORCH_LIBRARIES}
    gflags::gflags
    glog::glog
)
