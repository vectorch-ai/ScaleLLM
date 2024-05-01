include(pybind11_module)

pybind11_module(
  NAME
    wrapper
  HDRS
    llm.h
  SRCS
    llm.cpp
    scalellm.cpp
  DEPS
    :engine
    torch
    absl::strings
    gflags::gflags
    glog::glog
    Python::Python
)
