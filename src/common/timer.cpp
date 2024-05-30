#include "timer.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>

namespace llm {

Timer::Timer() : start_(absl::Now()) {}

// reset the timer
void Timer::reset() { start_ = absl::Now(); }

// get the elapsed time in seconds
double Timer::elapsed_seconds() const {
  return absl::ToDoubleSeconds(absl::Now() - start_);
}

}  // namespace llm