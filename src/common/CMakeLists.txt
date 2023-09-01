include(cc_library)

cc_library(
  NAME 
    common
  HDRS
    metrics.h
    slice.h
  SRCS
  DEPS
    prometheus-cpp::core
)
