#pragma once

#include <stdint.h>

namespace llm {

static constexpr uint64_t kMicrosToNanos = 1000ULL;
static constexpr uint64_t kMillisToNanos = 1000ULL * 1000ULL;
static constexpr uint64_t kSecondsToMillis = 1000ULL;
static constexpr uint64_t kSecondsToMicros = 1000ULL * 1000ULL;
static constexpr uint64_t kSecondsToNanos = 1000ULL * 1000ULL * 1000ULL;

class Time {
 public:
  Time() = default;
  ~Time() = default;

  uint64_t now_nanos() const;
  uint64_t now_micros() const;
  uint64_t now_seconds() const;
  uint64_t now_millis() const;
  
  static Time* instance() {
    static Time time;
    return &time;
  }
};

} // namespace llm
