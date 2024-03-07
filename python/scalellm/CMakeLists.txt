
set(PYBIND_DIR ../../../third_party/pybind11)
add_subdirectory(${PYBIND_DIR} build)

pybind11_add_module(wrapper csrc/scalellm.cpp)
target_link_libraries(wrapper PUBLIC llm)
set_target_properties(wrapper PROPERTIES OUTPUT_NAME "wrapper" PREFIX "")
