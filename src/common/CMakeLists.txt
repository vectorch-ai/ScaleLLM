include(cc_library)
include(cc_test)

cc_library(
  NAME 
    common
  HDRS
    macros.h
    metrics.h
    slice.h
    tensor_helper.h
    concurrent_queue.h
    time.h
    threadpool.h
    pretty_print.h
    json_reader.h
  SRCS
    time.cpp
    threadpool.cpp
    pretty_print.cpp
    json_reader.cpp
  DEPS
    absl::strings
    prometheus-cpp::core
    nlohmann_json::nlohmann_json
)

