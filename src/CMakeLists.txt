include(cc_binary)

cc_binary(
  NAME 
    example
  SRCS 
    example.cpp
  DEPS
    :torch_utils
    ${TORCH_LIBRARIES}
    gflags::gflags
    glog::glog
)

add_subdirectory(tokenizer)
add_subdirectory(models)
add_subdirectory(common)
