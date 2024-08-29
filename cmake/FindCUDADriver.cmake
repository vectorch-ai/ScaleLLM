# Find the cuda driver libraries
#
#
# The following are set after configuration is done:
#  CUDA_DRIVER_FOUND
#  CUDA_DRIVER_INCLUDE_DIRS
#  CUDA_DRIVER_LIBRARIES
#

find_path(CUDA_DRIVER_INCLUDE_DIRS
  NAMES cuda.h
  HINTS
    $ENV{CUDA_HOME}
    $ENV{CUDA_PATH}
    $ENV{CUDA_TOOLKIT_ROOT_DIR}
    /usr/local/cuda
  PATH_SUFFIXES
    include
)

find_library(CUDA_DRIVER_LIBRARIES 
  NAMES cuda
  HINTS 
    /usr/local/cuda/lib64
    /usr/local/lib
    /usr/local/lib/x86_64-linux-gnu
    /lib/x86_64-linux-gnu
    /usr/lib/x86_64-linux-gnu
    # stubs allow build without requiring a physical gpu device
    /usr/local/cuda/lib64/stubs
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUDA_DRIVER
  REQUIRED_VARS CUDA_DRIVER_INCLUDE_DIRS CUDA_DRIVER_LIBRARIES
)

if(CUDA_DRIVER_FOUND)
  message(STATUS "Found CUDA driver (include: ${CUDA_DRIVER_INCLUDE_DIRS}, library: ${CUDA_DRIVER_LIBRARIES})")
  if(NOT TARGET CUDA::driver)
    add_library(CUDA::driver SHARED IMPORTED)
    set_target_properties(CUDA::driver PROPERTIES
      IMPORTED_LOCATION "${CUDA_DRIVER_LIBRARIES}"
      INTERFACE_INCLUDE_DIRECTORIES "${CUDA_DRIVER_INCLUDE_DIRS}"
    )
  endif()
endif()
