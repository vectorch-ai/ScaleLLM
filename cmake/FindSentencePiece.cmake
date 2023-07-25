# FindSentencePiece.cmake
#
# Use this module as:
# 
#   find_package(SentencePiece)
#   find_package(SentencePiece REQUIRED)
# 
# This module provides the following imported targets, if found:
# 
# SentencePiece::sentencepiece
#   The Google SentencePiece library
#

find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
  pkg_check_modules(SENTENCEPIECE QUIET sentencepiece)
endif()

find_path(SentencePiece_INCLUDE_DIR 
  NAMES sentencepiece_processor.h 
  PATH_SUFFIXES include
  HINTS ${SENTENCEPIECE_INCLUDE_DIRS}
)
mark_as_advanced(SentencePiece_INCLUDE_DIR)

find_library(SentencePiece_LIBRARY 
  NAMES sentencepiece
  PATH_SUFFIXES lib
  HINTS ${SENTENCEPIECE_LIBRARY_DIRS}
)
mark_as_advanced(SentencePiece_LIBRARY)


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  SentencePiece
  DEFAULT_MSG
    SentencePiece_LIBRARY
    SentencePiece_INCLUDE_DIR
)

if(NOT SentencePiece_FOUND)
  if(SentencePiece_FIND_REQUIRED)
    message(FATAL_ERROR "Cannot find SentencePiece library")
  else()
    message(WARNING "SentencePiece library is not found!")
  endif()
else()
  if(SentencePiece_FOUND AND NOT TARGET SentencePiece::sentencepiece)
    add_library(SentencePiece::sentencepiece UNKNOWN IMPORTED)
    set_target_properties(SentencePiece::sentencepiece PROPERTIES
      IMPORTED_LOCATION "${SentencePiece_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${SentencePiece_INCLUDE_DIR}"
    )
  endif()
endif()
