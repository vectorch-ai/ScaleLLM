include(cc_binary)

cc_binary(
  NAME 
    mha_traits_viewer
  SRCS 
    mha_traits_viewer.cpp
  DEPS
    cutlass
    absl::strings
    absl::str_format
)