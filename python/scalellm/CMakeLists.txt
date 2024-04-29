
add_library(wrapper SHARED)
target_sources(wrapper PRIVATE csrc/scalellm.cpp)

target_link_libraries(wrapper PUBLIC llm)
set_target_properties(wrapper PROPERTIES OUTPUT_NAME "wrapper" PREFIX "")
