include(cc_library)
include(cc_test)

cc_library(
  NAME 
    common
  HDRS
    macros.h
    metrics.h
    timer.h
    slice.h
    scope_guard.h
    tensor_helper.h
    concurrent_queue.h
    threadpool.h
    pretty_print.h
    json_reader.h
  SRCS
    timer.cpp
    threadpool.cpp
    pretty_print.cpp
    json_reader.cpp
  DEPS
    absl::strings
    prometheus-cpp::core
    nlohmann_json::nlohmann_json
)

cc_test(
  NAME 
    common_test
  SRCS
    range_test.cpp
    threadpool_test.cpp
  DEPS
    common
    absl::synchronization
    absl::time
    GTest::gtest_main

)

