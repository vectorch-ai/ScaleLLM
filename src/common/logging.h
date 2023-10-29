#pragma once

#include <glog/logging.h>

namespace llm {

// define a list of macros for logging to avoid conflicts between glog and torch
#define GLOG(severity) COMPACT_GOOGLE_LOG_##severity.stream()

// conditional / occasional logging
#define GLOG_IF(severity, condition) LOG_IF(severity, condition)
#define GLOG_EVERY_N(severity, n) LOG_EVERY_N(severity, n)
#define GLOG_IF_EVERY_N(severity, condition, n) \
  LOG_IF_EVERY_N(severity, condition, n)
#define GLOG_FIRST_N(severity, n) LOG_FIRST_N(severity, n)

// debug logging
#define DGLOG(severity) DLOG(severity)
#define DGLOG_IF(severity, condition) DLOG_IF(severity, condition)
#define DGLOG_EVERY_N(severity, n) DLOG_EVERY_N(severity, n)

// verbose logging
#define VGLOG(severity) VLOG(severity)
#define VGLOG_IF(severity, condition) VLOG_IF(severity, condition)
#define VGLOG_EVERY_N(severity, n) VLOG_EVERY_N(severity, n)
#define VGLOG_IF_EVERYN(severity, condition, n) \
  VLOG_IF_EVERY_N(severity, condition, n)

}  // namespace llm