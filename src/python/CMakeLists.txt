include(cc_binary)
include(cc_library)

cmake_minimum_required(VERSION 3.5...3.26)
project(gen_py_wrappers LANGUAGES CXX)

add_compile_options(-fPIC)

set(PYBIND_DIR ../../third_party/pybind11)

add_subdirectory(${PYBIND_DIR} build)

pybind11_add_module(gen_py_wrappers py_wrappers.cpp)

target_link_libraries(gen_py_wrappers PUBLIC llm)
set_target_properties(gen_py_wrappers PROPERTIES OUTPUT_NAME "gen_py_wrappers" PREFIX "")

#set_target_properties(gen_py_wrappers
#  PROPERTIES
#  CXX_STANDARD 17
#  CXX_STANDARD_REQUIRED ON
#  PREFIX ""
#  SUFFIX ".so"
#)

