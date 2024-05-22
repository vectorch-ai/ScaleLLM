include(pybind_extension)

pybind_extension(
  NAME
    _C
  COPTS
    -DPY_MODULE_NAME=_C
  SRCS
    module.cpp
  DEPS
    :llm_handler
    torch
    absl::strings
    gflags::gflags
    glog::glog
    Python::Module
)
