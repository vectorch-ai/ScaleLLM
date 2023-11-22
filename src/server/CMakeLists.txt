include(cc_binary)
include(cc_library)

cc_library(
  NAME 
    grpc_server
  HDRS 
    call_data.h
    grpc_server.h
  SRCS 
    grpc_server.cpp
  DEPS
    :handlers
    :request
    absl::strings
    glog::glog
    grpc_proto::completion
)

cc_library(
  NAME
    http_server
  HDRS
    http_server.h
  SRCS
    http_server.cpp
  DEPS
    glog::glog
)

cc_binary(
  NAME 
    simple
  SRCS 
    simple.cpp
  DEPS
    :engine
    :model_downloader
    torch
    absl::strings
    gflags::gflags
    glog::glog
)

cc_binary(
  NAME 
    scalellm
  SRCS 
    main.cpp
  DEPS
    :grpc_server
    :http_server
    :engine
    absl::strings
    gflags::gflags
    glog::glog
    Folly::folly
    nlohmann_json::nlohmann_json
)

# install scalellm binary
install(TARGETS scalellm RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
# Install all dependencies for scalellm
install(CODE [[
    file(GET_RUNTIME_DEPENDENCIES
          RESOLVED_DEPENDENCIES_VAR DEPENDENCIES
          UNRESOLVED_DEPENDENCIES_VAR UNRESOLVED_DEPENDENCIES
          EXECUTABLES $<TARGET_FILE:scalellm>
          POST_EXCLUDE_REGEXES "^/lib(64)?/.*")

    file(INSTALL
          DESTINATION "${CMAKE_INSTALL_PREFIX}/lib"
          FILES ${DEPENDENCIES}
          FOLLOW_SYMLINK_CHAIN)

    # This should not be possible, but error out when a dependency cannot
    # be resolved.
    list(LENGTH UNRESOLVED_DEPENDENCIES UNRESOLVED_LENGTH)
    if(${UNRESOLVED_LENGTH} GREATER 0)
        message(FATAL_ERROR "Unresolved dependencies: ${UNRESOLVED_DEPENDENCIES}")
    endif()
]])

cc_binary(
  NAME 
    grpc_client
  HDRS 
  SRCS 
    grpc_client.cpp
  DEPS
    glog::glog
    gflags::gflags
    grpc_proto::completion
)

add_subdirectory(handlers)