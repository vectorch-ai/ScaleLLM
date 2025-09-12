cmake_minimum_required(VERSION 3.26)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

option(USE_CCACHE "Attempt using CCache to wrap the compilation" ON)
option(USE_MANYLINUX "Build for manylinux" OFF)

option(BUILD_NVBENCH "Build the nvbench binary" OFF)
option(INSTALL_PY_MODULE "Install python module to scalellm directory" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_COLOR_DIAGNOSTICS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MESSAGE_LOG_LEVEL STATUS)
set(CMAKE_VERBOSE_MAKEFILE ON)

if(POLICY CMP0135)
  # CMP0135: ExternalProject ignores timestamps in archives by default for the URL download method.
  cmake_policy(SET CMP0135 NEW)
endif()

function(parse_make_options options prefix)
  foreach(option ${options})
    string(REGEX REPLACE "(-D|-)" "" option ${option})
    string(REPLACE "=" ";" option ${option})
    list(GET option 0 option_name)
    list(GET option 1 option_value)
    set(${prefix}_${option_name}
        ${option_value}
        PARENT_SCOPE)
  endforeach()
endfunction()

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build type not set - defaulting to Release")
  set(CMAKE_BUILD_TYPE "Release"
      CACHE STRING "Choose the type of build from: Debug Release RelWithDebInfo MinSizeRel Coverage."
      FORCE
  )
endif()

if(USE_CCACHE)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "C compiler launcher")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "CXX compiler launcher")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "CUDA compiler launcher")
    message(STATUS "Using ccache: ${CCACHE_PROGRAM}")
    if (DEFINED ENV{CCACHE_DIR})
      message(STATUS "Using CCACHE_DIR: $ENV{CCACHE_DIR}")
    endif()
  else()
    message(WARNING "Could not find ccache. Consider installing ccache to speed up compilation.")
  endif()
endif()

# if defined, create and use the default binary cache for vcpkg
if (DEFINED ENV{VCPKG_DEFAULT_BINARY_CACHE})
  file(MAKE_DIRECTORY $ENV{VCPKG_DEFAULT_BINARY_CACHE})
  message(STATUS "Using VCPKG_DEFAULT_BINARY_CACHE: $ENV{VCPKG_DEFAULT_BINARY_CACHE}")
endif()

if (DEFINED ENV{DEPENDENCES_ROOT})
  message(STATUS "Using DEPENDENCES_ROOT: $ENV{DEPENDENCES_ROOT}")
endif()

# set architecture for CUDA
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# Build TORCH_CUDA_ARCH_LIST
set(TORCH_CUDA_ARCH_LIST "")
foreach(CUDA_ARCH IN LISTS CMAKE_CUDA_ARCHITECTURES)
  if(CUDA_ARCH MATCHES "^([1-9][0-9]*)([0-9]a?)$")
    set(TORCH_ARCH "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}")
  elseif(CUDA_ARCH STREQUAL "native")
    set(TORCH_ARCH "Auto")
  else()
    message(FATAL_ERROR "${CUDA_ARCH} is not supported")
  endif()
  list(APPEND TORCH_CUDA_ARCH_LIST ${TORCH_ARCH})
endforeach()

message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}")

# configure vcpkg
# have to set CMAKE_TOOLCHAIN_FILE before first project call.
if (DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
      CACHE STRING "Vcpkg toolchain file")
  message(STATUS "VCPKG_ROOT found, using vcpkg at $ENV{VCPKG_ROOT}")
else()
  include(FetchContent)
  if (DEFINED ENV{DEPENDENCES_ROOT})
    set(VCPKG_SOURCE_DIR $ENV{DEPENDENCES_ROOT}/vcpkg-src)
  else()
    set(VCPKG_SOURCE_DIR ${FETCHCONTENT_BASE_DIR}/vcpkg-src)
  endif()

  FetchContent_Declare(vcpkg
    GIT_REPOSITORY "https://github.com/microsoft/vcpkg.git"
    GIT_TAG "2025.04.09"
    SOURCE_DIR ${VCPKG_SOURCE_DIR}
  )
  FetchContent_MakeAvailable(vcpkg)

  message(STATUS "Downloading and using vcpkg at ${vcpkg_SOURCE_DIR}")
  set(CMAKE_TOOLCHAIN_FILE ${vcpkg_SOURCE_DIR}/scripts/buildsystems/vcpkg.cmake
      CACHE STRING "Vcpkg toolchain file")
endif()

project(
  "ScaleLLM"
  LANGUAGES C CXX CUDA
)

find_package(CUDAToolkit REQUIRED)

# setup CMake module path, defines path for include() and find_package()
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
enable_language(Rust)
find_package(Rust REQUIRED)

# include custom cmake modules
include(static_analyzers)
# TODO: can't use sanitizers with CUDA for now.
# include(sanitizers)

if(UNIX)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Og")
endif()

find_package(Boost REQUIRED)
find_package(Threads REQUIRED)
# find all dependencies from vcpkg
find_package(fmt CONFIG REQUIRED GLOBAL)
find_package(glog CONFIG REQUIRED)
find_package(gflags CONFIG REQUIRED)
find_package(absl CONFIG REQUIRED)
find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)
find_package(re2 CONFIG REQUIRED)
find_package(folly CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)
find_package(benchmark CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(prometheus-cpp CONFIG REQUIRED)
find_package(RapidJSON CONFIG REQUIRED)

if (USE_MANYLINUX)
  # manylinux doesn't ship Development.Embed
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
else()
  find_package(Python REQUIRED COMPONENTS Interpreter Development)
endif()

find_package(NCCL REQUIRED)

find_package(Jemalloc)
if(Jemalloc_FOUND)
  link_libraries(Jemalloc::jemalloc)
endif()

# Important Note: Always invoke find_package for other dependencies
# before including libtorch, as doing so afterwards may lead to
# unexpected linker errors.
if (DEFINED ENV{LIBTORCH_ROOT})
  find_package(Torch REQUIRED HINTS "$ENV{LIBTORCH_ROOT}")
  message(STATUS "Using libtorch at $ENV{LIBTORCH_ROOT}")
else()
  include(FetchContent)
  if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.9)
    # download libtorch 2.8.0 with cuda 12.9 from pytorch.org
    set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cu129/libtorch-shared-with-deps-2.8.0%2Bcu129.zip")
  elseif (CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.8)
    # download libtorch 2.8.0 with cuda 12.8 from pytorch.org
    set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cu128/libtorch-shared-with-deps-2.8.0%2Bcu128.zip")
  elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.6)
    # download libtorch 2.8.0 with cuda 12.6 from pytorch.org
    set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cu126/libtorch-shared-with-deps-2.8.0%2Bcu126.zip")
  else()
    # error out if cuda version is not supported
    message(FATAL_ERROR "Unsupported CUDA version: ${CUDAToolkit_VERSION}")
  endif()

  if (DEFINED ENV{DEPENDENCES_ROOT})
    set(LIBTORCH_SOURCE_DIR $ENV{DEPENDENCES_ROOT}/libtorch-src)
  else()
    set(LIBTORCH_SOURCE_DIR ${FETCHCONTENT_BASE_DIR}/libtorch-src)
  endif()

  FetchContent_Declare(libtorch
    URL ${LIBTORCH_URL}
    SOURCE_DIR ${LIBTORCH_SOURCE_DIR}
  )
  FetchContent_MakeAvailable(libtorch)

  find_package(Torch REQUIRED PATHS ${libtorch_SOURCE_DIR} NO_DEFAULT_PATH)
  message(STATUS "Downloading and using libtorch 2.8.0 for cuda ${CUDAToolkit_VERSION} at ${libtorch_SOURCE_DIR}")
endif()

# carry over torch flags to the rest of the project
message(STATUS "TORCH_CXX_FLAGS: ${TORCH_CXX_FLAGS}")
add_compile_options(${TORCH_CXX_FLAGS})
add_compile_definitions(TORCH_CUDA=1)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DC10_USE_GLOG")

message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")

# The following definitions must be undefined since half-precision operation is required.
list(APPEND CUDA_NVCC_FLAGS
      -U__CUDA_NO_HALF_OPERATORS__
      -U__CUDA_NO_HALF_CONVERSIONS__
      -U__CUDA_NO_HALF2_OPERATORS__
      -U__CUDA_NO_BFLOAT16_CONVERSIONS__
      -O3
      --use_fast_math
      -Xfatbin
      -compress-all)

# Enable aggresive fatbin compress for CUDA 12.8 or later.
if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.8)
  list(APPEND CUDA_NVCC_FLAGS -compress-mode=size)
endif()
message(STATUS "CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")

# enable testing in this directory so we can do a top-level `make test`.
# this also includes the BUILD_TESTING option, which is on by default.
include(CTest)
include(GoogleTest)

# include current path
list(APPEND COMMON_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src)
list(APPEND COMMON_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

# add subdirectories
add_subdirectory(proto)
add_subdirectory(src)
add_subdirectory(third_party)
add_subdirectory(scalellm)
