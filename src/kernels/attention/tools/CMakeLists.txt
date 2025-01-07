include(cc_binary)

cc_binary(
  NAME 
    attention_traits_viewer
  SRCS 
    attention_traits_viewer.cpp
  DEPS
    cutlass
    absl::strings
    absl::str_format
)