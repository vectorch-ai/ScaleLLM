include(cc_library)

cc_library(
  NAME 
    common
  HDRS
    metrics.h
    slice.h
    concurrent_queue.h
    executor.h
  SRCS
    executor.cpp
  DEPS
    prometheus-cpp::core
)
