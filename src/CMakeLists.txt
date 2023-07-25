include(cc_binary)

cc_binary(
  NAME 
    example
  SRCS 
    example.cpp
  DEPS
    ${TORCH_LIBRARIES}
)

add_subdirectory(tokenizer)
