#include "common/time.h"

#include <time.h>

namespace llm {

uint64_t Time::now_nanos() const {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts); 
  return (static_cast<uint64_t>(ts.tv_sec) * kSecondsToNanos +
          static_cast<uint64_t>(ts.tv_nsec));
}

uint64_t Time::now_micros() const {
  return now_nanos() / kMicrosToNanos;
}

uint64_t Time::now_seconds() const {
  return now_nanos() / kSecondsToNanos;
}

uint64_t Time::now_millis() const {
  return now_nanos() / kMillisToNanos;
}

} // namespace llm
