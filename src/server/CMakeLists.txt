include(cc_binary)

cc_binary(
  NAME 
    scalellm
  SRCS 
    main.cpp
  DEPS
    :torch_utils
    :llama
    :tokenizer
    ${TORCH_LIBRARIES}
    gflags::gflags
    glog::glog
)
