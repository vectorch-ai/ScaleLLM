include(cc_binary)
include(nvbench_binary)
include(cc_library)
include(cc_test)

cc_library(
  NAME
    attention.template
  HDRS
    fast_math.h
    layout_convertor.h
    fast_cast.cuh
    online_softmax.cuh
    safe_copy.h
    mask.h
    static_dispatch.h
    mha_params.h
    mha_tile.h
    mla_params.h
    mla_tile.h
    attn_combine_kernel.cuh
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
# globbing all generated files in sub directory "gensrc"
file(GLOB GENERATED_SRC_FILES "${CMAKE_CURRENT_BINARY_DIR}/gensrc/*.cu")

cc_library(
  NAME
    attention.kernels
  HDRS
    attn_api.h
  SRCS
    attn_api.cpp
    ${GENERATED_SRC_FILES}
  INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}
  DEPS
    :attention.template
    glog::glog
)

cc_test(
  NAME
    mha_kernel_test
  SRCS
    # mha_cpu_test.cpp
    sm80_mha_test.cu
    sm80_mha_pagedkv_test.cu
  DEPS
    :attention.template
    absl::random_random
    GTest::gtest_main
    torch
)

cc_test(
  NAME
    mla_kernel_test
  SRCS
    sm80_mla_test.cu
    sm80_mla_pagedkv_test.cu
  DEPS
    :attention.template
    absl::random_random
    GTest::gtest_main
    torch
)

cc_test(
  NAME
    attn_combine_kernel_test
  SRCS
    attn_combine_kernel_test.cu
  DEPS
    :attention.template
    absl::random_random
    GTest::gtest_main
    torch
)

nvbench_binary(
  NAME
    sm80_mha_bench
  SRCS
    sm80_mha_bench.cu
  DEPS
    :attention.template
)

nvbench_binary(
  NAME
    sm80_mha_pagedkv_bench
  SRCS
    sm80_mha_pagedkv_bench.cu
  DEPS
    absl::random_random
    :attention.template
)

nvbench_binary(
  NAME
    sm80_mla_bench
  SRCS
    mla_sm80_bench.cu
  DEPS
    :attention.template
)

add_subdirectory(tools)
