# FindJemalloc
# --------------
# 
# Use this module as:
# 
#   find_package(Jemalloc)
#   find_package(Jemalloc REQUIRED)
#
# This module provides the following imported targets, if found:
# 
# Jemalloc::jemalloc
#   The jemalloc library
#
find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
  pkg_check_modules(JEMALLOC QUIET jemalloc)
endif()

find_path(Jemalloc_INCLUDE_DIR 
  NAMES jemalloc/jemalloc.h 
  PATH_SUFFIXES include
  HINTS ${JEMALLOC_INCLUDE_DIRS}
)
mark_as_advanced(Jemalloc_INCLUDE_DIR)

set(JEMALLOC_SHARED_LIB_NAME
  "${CMAKE_SHARED_LIBRARY_PREFIX}jemalloc${CMAKE_SHARED_LIBRARY_SUFFIX}"
)

find_library(Jemalloc_LIBRARY 
  NAMES ${JEMALLOC_SHARED_LIB_NAME}
  PATH_SUFFIXES lib
  HINTS ${JEMALLOC_LIBRARY_DIRS}
)
mark_as_advanced(Jemalloc_LIBRARY)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Jemalloc
  DEFAULT_MSG
    Jemalloc_LIBRARY
    Jemalloc_INCLUDE_DIR
)

if(Jemalloc_FOUND)
  if(NOT TARGET Jemalloc::jemalloc)
    add_library(Jemalloc::jemalloc SHARED IMPORTED)
    set_target_properties(Jemalloc::jemalloc PROPERTIES
      IMPORTED_LOCATION "${Jemalloc_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${Jemalloc_INCLUDE_DIR}"
      INTERFACE_LINK_LIBRARIES "m;stdc++;Threads::Threads;dl"
    )
  endif()
endif()
