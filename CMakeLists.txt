cmake_minimum_required(VERSION 3.18)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# add libtorch to CMAKE_PREFIX_PATH
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libtorch")

# configure vcpkg
# have to set CMAKE_TOOLCHAIN_FILE before first project call.
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/.vcpkg/scripts/buildsystems/vcpkg.cmake"
  CACHE STRING "Vcpkg toolchain file")

project(
  "ScaleLLM"
  LANGUAGES CXX CUDA
)

# CMake 3.18 adds integrated support for architecture selection, disable it for now.
set(CMAKE_CUDA_ARCHITECTURES OFF)

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

# pybind11 requires python headers
find_package(Python REQUIRED COMPONENTS Development)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# find all dependencies from vcpkg
find_package(glog CONFIG REQUIRED)
find_package(gflags CONFIG REQUIRED)
find_package(absl CONFIG REQUIRED)
find_package(folly CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)
find_package(SentencePiece REQUIRED)

find_package(Jemalloc REQUIRED)
link_libraries(Jemalloc::jemalloc)

# enable testing in this directory so we can do a top-level `make test`.
# this also includes the BUILD_TESTING option, which is on by default.
include(CTest)
include(GoogleTest)

# include current path
list(APPEND COMMON_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src)

# add subdirectories
add_subdirectory(src)
add_subdirectory(third_party)
