#pragma once
#include <cstdint>
namespace llm {

// SamplingParameter is used to specify sampling parameters for a
// request.
struct SamplingParameter_ {
  float frequency_penalty = 0.0;
  float presence_penalty = 0.0;
  float repetition_penalty = 1.0;
  float temperature = 1.0;
  float top_p = 1.0;
  int64_t top_k = 0;
};

}  // namespace llm