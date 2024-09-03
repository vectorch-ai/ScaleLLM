# Find the CUDA driver libraries
#
#
# The following are set after configuration is done:
#  CUDADriver_FOUND
#  CUDADriver_INCLUDE_DIR
#  CUDADriver_LIBRARY
#

find_path(CUDADriver_INCLUDE_DIR
  NAMES cuda.h
  HINTS
    $ENV{CUDA_HOME}
    $ENV{CUDA_PATH}
    $ENV{CUDA_TOOLKIT_ROOT_DIR}
    /usr/local/cuda
  PATH_SUFFIXES
    include
)

mark_as_advanced(CUDADriver_INCLUDE_DIR)

find_library(CUDADriver_LIBRARY 
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

mark_as_advanced(CUDADriver_LIBRARY)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUDADriver
  REQUIRED_VARS CUDADriver_INCLUDE_DIR CUDADriver_LIBRARY
)

if(CUDADriver_FOUND)
  message(STATUS "Found CUDADriver : ${CUDADriver_LIBRARY}")
  if(NOT TARGET CUDA::driver)
    add_library(CUDA::driver SHARED IMPORTED)
    set_target_properties(CUDA::driver PROPERTIES
      IMPORTED_LOCATION "${CUDADriver_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${CUDADriver_INCLUDE_DIR}"
    )
  endif()
endif()
