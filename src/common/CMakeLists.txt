include(cc_library)
include(cc_test)

cc_library(
  NAME 
    common
  HDRS
    logging.h
    metrics.h
    slice.h
    concurrent_queue.h
    executor.h
    pretty_print.h
    json_reader.h
  SRCS
    executor.cpp
    pretty_print.cpp
    json_reader.cpp
  DEPS
    absl::strings
    prometheus-cpp::core
    nlohmann_json::nlohmann_json
)

cc_library(
  NAME 
    process_group
  HDRS
    process_group.h
  SRCS
    process_group.cpp
  DEPS
    torch
    NCCL::nccl
    glog::glog
)

cc_test(
  NAME
    process_group_test
  SRCS
    process_group_test.cpp
  DEPS
    :process_group
    GTest::gtest_main
)

