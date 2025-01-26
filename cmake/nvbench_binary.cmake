include(CMakeParseArguments)

function(nvbench_binary)
  if(NOT BUILD_NVBENCH)
    return()
  endif()

  cmake_parse_arguments(
    NV_BINARY # prefix
    "" # options
    "NAME" # one value args
    "HDRS;SRCS;COPTS;DEFINES;LINKOPTS;DEPS" # multi value args
    ${ARGN}
  )

  add_executable(${NV_BINARY_NAME} "")
  target_sources(${NV_BINARY_NAME} 
    PRIVATE ${NV_BINARY_SRCS} ${NV_BINARY_HDRS}
  )
  target_link_libraries(${NV_BINARY_NAME}
    PUBLIC 
      ${NV_BINARY_DEPS}
      nvbench::nvbench
      nvbench::main
    PRIVATE
      ${NV_BINARY_LINKOPTS}
  )
  target_include_directories(${NV_BINARY_NAME}
    PUBLIC
      "$<BUILD_INTERFACE:${COMMON_INCLUDE_DIRS}>"      
  )
  target_compile_options(${NV_BINARY_NAME} 
    PRIVATE 
      ${NV_BINARY_COPTS}
      -lineinfo
  )
  target_compile_definitions(${NV_BINARY_NAME} PUBLIC ${NV_BINARY_DEFINES})

  add_executable(:${NV_BINARY_NAME} ALIAS ${NV_BINARY_NAME})
endfunction()
