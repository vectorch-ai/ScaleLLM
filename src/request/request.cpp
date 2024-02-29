#include "request.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>

#include <cstdint>
#include <string>
#include <vector>

#include "common/logging.h"
#include "sequence.h"

namespace llm {

Request::Request(const std::string& id,
                 const std::vector<int32_t>& prompt_tokens)
    : id(id),
      created_time(absl::ToUnixSeconds(absl::Now())),
      prompt_tokens(prompt_tokens) {}

void Request::add_sequence(OnStream on_stream) {
  if (stream) {
    GCHECK(on_stream) << "on_stream should not be null if stream is true";
  }
  sequences.emplace_back(this->sampling_param,
                         this->stopping_criteria,
                         this->prompt_tokens,
                         this->echo,
                         on_stream);
}

bool Request::is_finished() const {
  for (const auto& seq : sequences) {
    if (!seq.is_finished()) {
      return false;
    }
  }
  return true;
}
}  // namespace llm
