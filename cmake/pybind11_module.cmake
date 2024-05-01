include(CMakeParseArguments)

# pybind11_module()
#
# Parameters:
# NAME: name of module
# HDRS: List of public header files for the library
# SRCS: List of source files for the library
# DEPS: List of other libraries to be linked in to the binary targets
# COPTS: List of private compile options
# DEFINES: List of public defines
# LINKOPTS: List of link options
#
# pybind11_module(
#   NAME
#     awesome
#   HDRS
#     "a.h"
#   SRCS
#     "a.cc"
# )
#

if(NOT DEFINED PYTHON_MODULE_EXTENSION OR NOT DEFINED PYTHON_MODULE_DEBUG_POSTFIX)
  execute_process(
    COMMAND
      "${Python_EXECUTABLE}" "-c"
      "import sys, importlib; s = importlib.import_module('distutils.sysconfig' if sys.version_info < (3, 10) else 'sysconfig'); print(s.get_config_var('EXT_SUFFIX') or s.get_config_var('SO'))"
    OUTPUT_VARIABLE _PYTHON_MODULE_EXT_SUFFIX
    ERROR_VARIABLE _PYTHON_MODULE_EXT_SUFFIX_ERR
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(_PYTHON_MODULE_EXT_SUFFIX STREQUAL "")
    message(
      FATAL_ERROR "pybind11 could not query the module file extension, likely the 'distutils'"
                  "package is not installed. Full error message:\n${_PYTHON_MODULE_EXT_SUFFIX_ERR}"
    )
  endif()

  # This needs to be available for the pybind11_extension function
  if(NOT DEFINED PYTHON_MODULE_DEBUG_POSTFIX)
    get_filename_component(_PYTHON_MODULE_DEBUG_POSTFIX "${_PYTHON_MODULE_EXT_SUFFIX}" NAME_WE)
    set(PYTHON_MODULE_DEBUG_POSTFIX
        "${_PYTHON_MODULE_DEBUG_POSTFIX}"
        CACHE INTERNAL "")
  endif()

  if(NOT DEFINED PYTHON_MODULE_EXTENSION)
    get_filename_component(_PYTHON_MODULE_EXTENSION "${_PYTHON_MODULE_EXT_SUFFIX}" EXT)
    set(PYTHON_MODULE_EXTENSION
        "${_PYTHON_MODULE_EXTENSION}"
        CACHE INTERNAL "")
  endif()
endif()

function(pybind11_module)
  cmake_parse_arguments(
    PYBIND11 # prefix
    "TESTONLY" # options
    "NAME" # one value args
    "HDRS;SRCS;COPTS;DEFINES;LINKOPTS;DEPS" # multi value args
    ${ARGN}
  )

  if(PYBIND11_TESTONLY AND (NOT BUILD_TESTING))
    return()
  endif()

  add_library(${PYBIND11_NAME} SHARED)
  target_sources(${PYBIND11_NAME} 
    PRIVATE ${PYBIND11_SRCS} ${PYBIND11_HDRS})
  target_link_libraries(${PYBIND11_NAME}
    PUBLIC ${PYBIND11_DEPS}
    PRIVATE ${PYBIND11_LINKOPTS}
  )
  target_compile_options(${PYBIND11_NAME} PRIVATE ${PYBIND11_COPTS})
  target_compile_definitions(${PYBIND11_NAME} PUBLIC ${PYBIND11_DEFINES})

  # -fvisibility=hidden is required to allow multiple modules compiled against
  # different pybind versions to work properly, and for some features (e.g.
  # py::module_local). 
  if(NOT DEFINED CMAKE_CXX_VISIBILITY_PRESET)
    set_target_properties(${PYBIND11_NAME} PROPERTIES CXX_VISIBILITY_PRESET "hidden")
  endif()

  if(NOT DEFINED CMAKE_CUDA_VISIBILITY_PRESET)
    set_target_properties(${PYBIND11_NAME} PROPERTIES CUDA_VISIBILITY_PRESET "hidden")
  endif()

  set_target_properties(
    ${PYBIND11_NAME}
    PROPERTIES PREFIX ""
               DEBUG_POSTFIX "${PYTHON_MODULE_DEBUG_POSTFIX}"
               SUFFIX "${PYTHON_MODULE_EXTENSION}")

endfunction()
