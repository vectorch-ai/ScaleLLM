# Find the nccl libraries
#
# The following are set after configuration is done:
#  NCCL_FOUND
#  NCCL_INCLUDE_DIR
#  NCCL_LIBRARY
#
# Adapted from https://github.com/pytorch/pytorch/blob/master/cmake/Modules/FindNCCL.cmake
#
set(NCCL_VERSION $ENV{NCCL_VERSION} CACHE STRING "Version of NCCL to build with")

list(APPEND NCCL_ROOT $ENV{NCCL_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR})
# Compatible layer for CMake <3.12. NCCL_ROOT will be accounted in for searching paths and libraries for CMake >=3.12.
list(APPEND CMAKE_PREFIX_PATH ${NCCL_ROOT})

find_path(NCCL_INCLUDE_DIR
  NAMES nccl.h
  HINTS
    $ENV{NCCL_ROOT}
    $ENV{CUDA_HOME}
    $ENV{CUDA_PATH}
    $ENV{CUDA_TOOLKIT_ROOT_DIR}
    $ENV{NCCL}
    /usr/local/cuda
    /usr
  PATH_SUFFIXES
    include
)
mark_as_advanced(NCCL_INCLUDE_DIR)

if (USE_STATIC_NCCL)
  MESSAGE(STATUS "USE_STATIC_NCCL is set. Linking with static NCCL library.")
  SET(NCCL_LIBNAME "nccl_static")
  if (NCCL_VERSION)  # Prefer the versioned library if a specific NCCL version is specified
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a.${NCCL_VERSION}" ${CMAKE_FIND_LIBRARY_SUFFIXES})
  endif()
else()
  SET(NCCL_LIBNAME "nccl")
  if (NCCL_VERSION)  # Prefer the versioned library if a specific NCCL version is specified
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".so.${NCCL_VERSION}" ${CMAKE_FIND_LIBRARY_SUFFIXES})
  endif()
endif()

# Read version from header
if(EXISTS "${NCCL_INCLUDE_DIR}/nccl.h")
  file(READ ${NCCL_INCLUDE_DIR}/nccl.h NCCL_HEADER_CONTENTS)
endif()

if(NCCL_HEADER_CONTENTS)
  string(REGEX MATCH "define NCCL_MAJOR * +([0-9]+)"
               NCCL_VERSION_MAJOR "${NCCL_HEADER_CONTENTS}")
  string(REGEX REPLACE "define NCCL_MAJOR * +([0-9]+)" "\\1"
               NCCL_VERSION_MAJOR "${NCCL_VERSION_MAJOR}")
  string(REGEX MATCH "define NCCL_MINOR * +([0-9]+)"
               NCCL_VERSION_MINOR "${NCCL_HEADER_CONTENTS}")
  string(REGEX REPLACE "define NCCL_MINOR * +([0-9]+)" "\\1"
    NCCL_VERSION_MINOR "${NCCL_VERSION_MINOR}")
  string(REGEX MATCH "define NCCL_PATCH * +([0-9]+)"
    NCCL_VERSION_PATCH "${NCCL_HEADER_CONTENTS}")
  string(REGEX REPLACE "define NCCL_PATCH * +([0-9]+)" "\\1"
    NCCL_VERSION_PATCH "${NCCL_VERSION_PATCH}")
  if(NOT NCCL_VERSION_MAJOR)
    set(NCCL_VERSION "?")
  else()
    set(NCCL_VERSION "${NCCL_VERSION_MAJOR}.${NCCL_VERSION_MINOR}.${NCCL_VERSION_PATCH}")
  endif()
endif()

find_library(NCCL_LIBRARY
  NAMES ${NCCL_LIBNAME}
  HINTS
    $ENV{NCCL_ROOT}
    $ENV{CUDA_HOME}
    $ENV{CUDA_PATH}
    $ENV{CUDA_TOOLKIT_ROOT_DIR}
    /usr/local/cuda
    /usr/lib/x86_64-linux-gnu/
  PATH_SUFFIXES
    lib
    lib64
)
mark_as_advanced(NCCL_LIBRARY)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL
  REQUIRED_VARS NCCL_INCLUDE_DIR NCCL_LIBRARY
  VERSION_VAR   NCCL_VERSION)

if(NCCL_FOUND)
  message(STATUS "Found NCCL ${NCCL_VERSION} (include: ${NCCL_INCLUDE_DIRS}, library: ${NCCL_LIBRARIES})")
  if(NOT TARGET NCCL::nccl)
    add_library(NCCL::nccl UNKNOWN IMPORTED)
    set_target_properties(NCCL::nccl PROPERTIES
      IMPORTED_LOCATION "${NCCL_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIR}"
    )
  endif()
endif()
