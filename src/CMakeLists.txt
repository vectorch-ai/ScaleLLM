include(cc_binary)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

cc_binary(
  NAME 
    example
  SRCS 
    example.cpp
  DEPS
    ${TORCH_LIBRARIES}
)
