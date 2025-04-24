include(pybind_extension)

pybind_extension(
  NAME
    _C
  COPTS
    -DPY_MODULE_NAME=_C
  SRCS
    csrc/sampling_params.cpp
    csrc/output.cpp
    csrc/llm_handler.cpp
    csrc/module.cpp
    csrc/kernels.cu
  DEFINES
    PYBIND11_DETAILED_ERROR_MESSAGES=1
  LINKDIRS
    ${TORCH_INSTALL_PREFIX}/lib
  DEPS
    :llm_handler
    :marlin.kernels
    torch
    torch_python
    absl::strings
    gflags::gflags
    glog::glog
    Python::Module
)

if (INSTALL_PY_MODULE)
  add_custom_command(TARGET _C POST_BUILD
      COMMENT "Copying $<TARGET_FILE_NAME:_C> to scalellm directory"
      COMMAND ${CMAKE_COMMAND} -E create_symlink
              $<TARGET_FILE:_C>
              ${CMAKE_SOURCE_DIR}/scalellm/$<TARGET_FILE_NAME:_C>
  )
endif()
