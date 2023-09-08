include(CMakeParseArguments)
include(CMakePrintHelpers)

# inspired by https://github.com/abseil/abseil-cpp
# proto_library()
# CMake function to imitate Bazel's proto_library rule.
#
# Parameters:
# NAME: name of target
# SRCS: List of proto source files for the library
# DEPS: List of other libraries to be linked in to the binary targets
# COPTS: List of private compile options
# DEFINES: List of public defines
# LINKOPTS: List of link options
#
# cc_library(
#   NAME
#     awesome
#   HDRS
#     "a.h"
#   SRCS
#     "a.cc"
# )
# proto_library(
#   NAME
#     proto_lib
#   SRCS
#     "b.proto"
#   DEPS
#     :awesome
# )
#
function(proto_library)
  # parse arguments and set variables
  cmake_parse_arguments(
    PROTO_LIB # prefix
    "" # options
    "NAME" # one value args
    "SRCS;COPTS;DEFINES;LINKOPTS;DEPS" # multi value args
    ${ARGN}
  )
  # generate cpp and hpp files from proto files using protoc compiler
  protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_LIB_SRCS})

  add_library(${PROTO_LIB_NAME} STATIC)
  target_sources(${PROTO_LIB_NAME}
    PRIVATE ${PROTO_SRCS} ${PROTO_HDRS}
  )

  target_link_libraries(${PROTO_LIB_NAME}
    PUBLIC protobuf::libprotobuf
  )
  target_include_directories(${PROTO_LIB_NAME}
    PUBLIC 
      ${Protobuf_INCLUDE_DIRS}
      ${CMAKE_CURRENT_BINARY_DIR}
  )
  target_compile_options(${PROTO_LIB_NAME} 
    PRIVATE 
      ${PROTO_LIB_COPTS}
      -Wno-unused-parameter
  )
  target_compile_definitions(${PROTO_LIB_NAME} 
    PUBLIC 
      ${PROTO_LIB_DEFINES}
  )

  add_library(proto::${PROTO_LIB_NAME} ALIAS ${PROTO_LIB_NAME})
endfunction()
