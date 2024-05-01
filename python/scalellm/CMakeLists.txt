include(cc_library)
cc_library(
  NAME
    llm
  HDRS
    csrc/llm.h
  SRCS
    csrc/llm.cpp
  DEPS
    :engine
    torch
    absl::strings
    gflags::gflags
    glog::glog
    Python::Python
)

add_library(wrapper SHARED)
target_sources(wrapper PRIVATE csrc/scalellm.cpp)

target_link_libraries(wrapper PUBLIC llm)
set_target_properties(wrapper PROPERTIES OUTPUT_NAME "wrapper" PREFIX "")
