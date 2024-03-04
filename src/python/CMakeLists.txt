cmake_minimum_required(VERSION 3.5...3.26)
project(example LANGUAGES CXX)

set(PYBIND_DIR ../../third_party/pybind11)

add_subdirectory(${PYBIND_DIR} build)

pybind11_add_module(pybind pybind.cpp)

target_link_libraries(pybind PUBLIC ${LIBRARIES})
set_target_properties(pybind PROPERTIES OUTPUT_NAME "pybind" PREFIX "")

