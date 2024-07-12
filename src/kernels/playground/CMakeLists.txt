include(cc_binary)

cc_binary(
  NAME 
    tiled_mma
  SRCS 
    tiled_mma.cpp
  DEPS
    cutlass
)
