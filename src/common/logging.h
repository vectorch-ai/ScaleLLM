#pragma once

#include <glog/logging.h>

namespace llm {

// define a list of macros for logging to avoid conflicts between glog and torch
#define GLOG(severity) COMPACT_GOOGLE_LOG_##severity.stream()

// conditional / occasional logging
#define GLOG_IF(severity, condition) \
  static_cast<void>(0),              \
      !(condition) ? (void)0 : google::LogMessageVoidify() & GLOG(severity)

#define GLOG_EVERY_N(severity, n) \
  SOME_KIND_OF_LOG_EVERY_N(severity, (n), google::LogMessage::SendToLog)

#define GLOG_IF_EVERY_N(severity, condition, n) \
  SOME_KIND_OF_LOG_IF_EVERY_N(                  \
      severity, (condition), (n), google::LogMessage::SendToLog)

#define GLOG_FIRST_N(severity, n) \
  SOME_KIND_OF_LOG_FIRST_N(severity, (n), google::LogMessage::SendToLog)

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

// CHECK macros
#define GCHECK(condition)                                       \
  GLOG_IF(FATAL, GOOGLE_PREDICT_BRANCH_NOT_TAKEN(!(condition))) \
      << "Check failed: " #condition " "

#define GCHECK_EQ(val1, val2) CHECK_OP(_EQ, ==, val1, val2)
#define GCHECK_NE(val1, val2) CHECK_OP(_NE, !=, val1, val2)
#define GCHECK_LE(val1, val2) CHECK_OP(_LE, <=, val1, val2)
#define GCHECK_LT(val1, val2) CHECK_OP(_LT, <, val1, val2)
#define GCHECK_GE(val1, val2) CHECK_OP(_GE, >=, val1, val2)
#define GCHECK_GT(val1, val2) CHECK_OP(_GT, >, val1, val2)

// Check that the input is non NULL.  This very useful in constructor
// initializer lists.
#define GCHECK_NOTNULL(val) \
  google::CheckNotNull(__FILE__, __LINE__, "'" #val "' Must be non NULL", (val))

}  // namespace llm