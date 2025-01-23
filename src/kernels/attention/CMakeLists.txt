include(cc_binary)
include(cc_library)
include(cc_test)

cc_library(
  NAME 
    attention.template
  HDRS
    ptx.cuh
    fast_cast.cuh
    online_softmax.cuh
    mask.h
    static_dispatch.h
    attention_params.h
    attention_tile.h
    attention_traits_sm80.h
    attention_kernel_sm80.cuh
    attention_launch_sm80.cuh
  DEPS
    cutlass
)

# copy the file to binary dir in order to force re-configuration when the file changes
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/generate_instantiation_cu.py
  ${CMAKE_CURRENT_BINARY_DIR}/generate_instantiation_cu.py
)

# execute the script to generate the instantiation of the kernels
execute_process(
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/generate_instantiation_cu.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/
  COMMAND_ERROR_IS_FATAL ANY
)
# globbing all generated files in sub directory "generated"
file(GLOB GENERATED_SRC_FILES "${CMAKE_CURRENT_BINARY_DIR}/generated/*.cu")

cc_library(
  NAME 
    attention.kernel
  HDRS
    # attention.h
  SRCS
    # attention.cpp
    ${GENERATED_SRC_FILES}
  INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}
  DEPS
    :attention.template
    glog::glog
)

cc_test(
  NAME
    attention_kernel_test
  SRCS
    # attention_cpu_test.cpp
    attention_traits_test.cpp
    attention_kernel_sm80_test.cu
    attention_kernel_sm80_varlen_test.cu
    attention_kernel_sm80_pagedkv_test.cu
  DEPS
    :attention.template
    absl::random_random
    GTest::gtest_main
    torch
)

cc_binary(
  NAME 
    attention_bench_sm80
  SRCS 
    attention_bench_sm80.cu
  DEPS
    nvbench::nvbench
    nvbench::main
    :attention.template
  COPTS
    -lineinfo 
)

add_subdirectory(flash_attn)
add_subdirectory(flash_infer)
add_subdirectory(tools)