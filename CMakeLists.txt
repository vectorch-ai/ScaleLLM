cmake_minimum_required(VERSION 3.16)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# configure vcpkg
# have to set CMAKE_TOOLCHAIN_FILE before first project call.
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/.vcpkg/scripts/buildsystems/vcpkg.cmake"
  CACHE STRING "Vcpkg toolchain file")

project(
  "llm-serving"
  LANGUAGES CXX
)

# setup CMake module path, defines path for include() and find_package()
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# include custom cmake modules
include(static_analyzers)
include(sanitizers)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# find all dependencies from vcpkg
find_package(glog CONFIG REQUIRED)
find_package(gflags CONFIG REQUIRED)
find_package(absl CONFIG REQUIRED)

# enable testing in this directory so we can do a top-level `make test`.
# this also includes the BUILD_TESTING option, which is on by default.
include(CTest)

# add subdirectories
add_subdirectory(src)

