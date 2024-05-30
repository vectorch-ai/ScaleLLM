#pragma once

#include <absl/time/time.h>

namespace llm {

class Timer final {
 public:
  Timer();

  // reset the timer
  void reset();

  // get the elapsed time in seconds
  double elapsed() const;

 private:
  // the start time of the timer
  absl::Time start_;
};

}  // namespace llm