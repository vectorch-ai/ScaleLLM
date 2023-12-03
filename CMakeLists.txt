cmake_minimum_required(VERSION 3.18)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if(POLICY CMP0135)
  # CMP0135: ExternalProject ignores timestamps in archives by default for the URL download method.
  cmake_policy(SET CMP0135 NEW)
endif()

# if defined, create and use the default binary cache for vcpkg
if (DEFINED ENV{VCPKG_DEFAULT_BINARY_CACHE})
  set(VCPKG_DEFAULT_BINARY_CACHE "$ENV{VCPKG_DEFAULT_BINARY_CACHE}")
  file(MAKE_DIRECTORY ${VCPKG_DEFAULT_BINARY_CACHE})
  message(STATUS "Using VCPKG_DEFAULT_BINARY_CACHE: ${VCPKG_DEFAULT_BINARY_CACHE}")
endif()


# set architecture for CUDA
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)
  set(TORCH_CUDA_ARCH_LIST "8.0 8.6 8.9 9.0")
  message(STATUS "Using CUDA Architecture: ${CMAKE_CUDA_ARCHITECTURES}")
else()
  message(STATUS "Using CUDA Architecture: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# configure vcpkg
# have to set CMAKE_TOOLCHAIN_FILE before first project call.
if (DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
      CACHE STRING "Vcpkg toolchain file")
  message(STATUS "VCPKG_ROOT found, using vcpkg at $ENV{VCPKG_ROOT}")
else()
  include(FetchContent)
  FetchContent_Declare(vcpkg
    GIT_REPOSITORY "https://github.com/microsoft/vcpkg.git"
    GIT_TAG "2023.10.19"
  )
  FetchContent_MakeAvailable(vcpkg)

  message(STATUS "VCPKG_ROOT not found, downloading and using vcpkg at ${vcpkg_SOURCE_DIR}")
  set(CMAKE_TOOLCHAIN_FILE ${vcpkg_SOURCE_DIR}/scripts/buildsystems/vcpkg.cmake
      CACHE STRING "Vcpkg toolchain file")
endif()

project(
  "ScaleLLM"
  LANGUAGES CXX CUDA
)
find_package(CUDA REQUIRED)

# setup CMake module path, defines path for include() and find_package()
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
enable_language(Rust)
find_package(Rust REQUIRED)

# include custom cmake modules
include(static_analyzers)
# TODO: can't use sanitizers with CUDA for now.
# include(sanitizers)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Boost CONFIG REQUIRED)
find_package(Threads REQUIRED)
# find all dependencies from vcpkg
find_package(glog CONFIG REQUIRED)
find_package(gflags CONFIG REQUIRED)
find_package(absl CONFIG REQUIRED)
find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)
find_package(folly CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)
find_package(benchmark CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(prometheus-cpp CONFIG REQUIRED)
find_package(stduuid CONFIG REQUIRED)

set(USE_STATIC_NCCL ON)
find_package(NCCL REQUIRED)

# libtorch requires python headers
find_package(Python REQUIRED COMPONENTS Development)

find_package(Jemalloc)
if(Jemalloc_FOUND)
  message(STATUS "jemalloc found, linking to jemalloc")
  link_libraries(Jemalloc::jemalloc)
endif()

# Important Note: Always invoke find_package for other dependencies
# before including libtorch, as doing so afterwards may lead to
# unexpected linker errors.
if (DEFINED ENV{LIBTORCH_ROOT})
  message(STATUS "LIBTORCH_ROOT found, using libtorch at $ENV{LIBTORCH_ROOT}")
  find_package(Torch REQUIRED HINTS "$ENV{LIBTORCH_ROOT}")
else()
  include(FetchContent)
  if (CUDA_VERSION VERSION_GREATER_EQUAL 12.1)
    # download libtorch 2.1 with cuda 12.1 from pytorch.org
    set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip")
    message(STATUS "LIBTORCH_ROOT not found, downloading and using libtorch 2.1 for cuda ${CUDA_VERSION}")
  elseif(CUDA_VERSION VERSION_GREATER_EQUAL 11.8)
    # download libtorch 2.1 with cuda 11.8 from pytorch.org
    set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu118.zip")
    message(STATUS "LIBTORCH_ROOT not found, downloading and using libtorch 2.1 for cuda ${CUDA_VERSION}")
  else()
    # error out if cuda version is not supported
    message(FATAL_ERROR "Unsupported CUDA version: ${CUDA_VERSION}")
  endif()

  FetchContent_Declare(libtorch URL ${LIBTORCH_URL})
  FetchContent_MakeAvailable(libtorch)
  
  find_package(Torch REQUIRED PATHS ${libtorch_SOURCE_DIR} NO_DEFAULT_PATH)
endif()

# carry over torch flags to the rest of the project
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# enable testing in this directory so we can do a top-level `make test`.
# this also includes the BUILD_TESTING option, which is on by default.
include(CTest)
include(GoogleTest)

# include current path
list(APPEND COMMON_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR})
list(APPEND COMMON_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src)
list(APPEND COMMON_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

# add subdirectories
add_subdirectory(proto)
add_subdirectory(src)
add_subdirectory(third_party)
