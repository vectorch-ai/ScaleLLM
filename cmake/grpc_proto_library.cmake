include(CMakeParseArguments)
include(CMakePrintHelpers)

# inspired by https://github.com/abseil/abseil-cpp
# grpc_proto_library()
# CMake function to imitate Bazel's grpc_proto_library rule.
#
# Parameters:
# NAME: name of target
# SRCS: List of proto source files for the library
# DEPS: List of other libraries to be linked in to the binary targets
# COPTS: List of private compile options
# DEFINES: List of public defines
# LINKOPTS: List of link options
#
# grpc_proto_library(
#   NAME
#     proto_lib
#   SRCS
#     "b.proto"
# )
#
function(grpc_proto_library)
  cmake_parse_arguments(
    PROTO_LIB # prefix
    "" # options
    "NAME" # one value args
    "SRCS;COPTS;DEFINES;LINKOPTS;DEPS" # multi value args
    ${ARGN}
  )

  # Add Library target with protobuf sources
  add_library(${PROTO_LIB_NAME} ${PROTO_LIB_SRCS})

  # Link dependencies
  target_link_libraries(${PROTO_LIB_NAME}
    PUBLIC
      protobuf::libprotobuf
      gRPC::grpc
      gRPC::grpc++
      gRPC::grpc++_reflection
    PRIVATE
      ${PROTO_LIB_DEPS}
  )

  # Set include directories
  target_include_directories(${PROTO_LIB_NAME}
    PUBLIC 
      ${Protobuf_INCLUDE_DIRS}
      ${CMAKE_CURRENT_BINARY_DIR}
  )

  # Set compile options
  target_compile_options(${PROTO_LIB_NAME}
    PRIVATE
      ${PROTO_LIB_COPTS}
      -Wno-unused-parameter
  )

  # Set compile definitions
  target_compile_definitions(${PROTO_LIB_NAME}
    PUBLIC ${PROTO_LIB_DEFINES}
  )

  # Compile protobuf and grpc files
  protobuf_generate(TARGET ${PROTO_LIB_NAME} LANGUAGE cpp)

  # Get grpc_cpp_plugin location
  get_target_property(grpc_cpp_plugin_location gRPC::grpc_cpp_plugin LOCATION)

  # Generate grpc files from protobuf files using grpc_cpp_plugin
  protobuf_generate(
    TARGET ${PROTO_LIB_NAME}
    LANGUAGE grpc
    GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc
    PLUGIN "protoc-gen-grpc=${grpc_cpp_plugin_location}"
  )

  # Set alias for library
  add_library(grpc_proto::${PROTO_LIB_NAME} ALIAS ${PROTO_LIB_NAME})
endfunction()
